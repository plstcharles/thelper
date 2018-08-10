import json
import logging
import os
import time
import platform
from copy import deepcopy
from abc import abstractmethod

import torch
import torch.optim

from tensorboardX import SummaryWriter

import thelper.utils

logger = logging.getLogger(__name__)


def load_trainer(session_name, save_dir, config, model, loss,
                 metrics, optimizer, scheduler, loaders):
    if "trainer" not in config or not config["trainer"]:
        raise AssertionError("config missing 'trainer' field")
    trainer_config = config["trainer"]
    if "type" not in trainer_config or not trainer_config["type"]:
        raise AssertionError("trainer config missing 'type' field")
    trainer_type = thelper.utils.import_class(trainer_config["type"])
    if "params" not in trainer_config:
        raise AssertionError("trainer config missing 'params' field")
    params = thelper.utils.keyvals2dict(trainer_config["params"])
    metapack = (model, loss, metrics, optimizer, scheduler)
    trainer = trainer_type(session_name, save_dir, metapack, loaders, trainer_config, **params)
    return trainer


class Trainer:

    def __init__(self, session_name, save_dir, metapack, loaders, config):
        model, loss, metrics, optimizer, scheduler = metapack
        if not model or not loss or not metrics or not optimizer or not config:
            raise AssertionError("missing input args")
        train_loader, valid_loader, test_loader = loaders
        if not (train_loader or valid_loader or test_loader):
            raise AssertionError("must provide at least one loader with available data")
        self.logger = thelper.utils.get_class_logger()
        self.name = session_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        if "epochs" not in config or not config["epochs"] or int(config["epochs"]) <= 0:
            raise AssertionError("bad trainer config epoch count")
        if train_loader:
            self.epochs = int(config["epochs"])
        else:
            self.epochs = 1
            self.logger.info("no training data provided, will run a single epoch on valid/test data")
        self.save_freq = int(config["save_freq"]) if "save_freq" in config else 1
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        self.config_backup_path = os.path.join(self.save_dir, "config.json")  # this file is created in cli.get_save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.use_tbx = False
        if "use_tbx" in config:
            self.use_tbx = thelper.utils.str2bool(config["use_tbx"])
        writers = [None, None, None]
        if self.use_tbx:
            self.tbx_root_dir = os.path.join(self.save_dir, "tbx_logs")
            if not os.path.exists(self.tbx_root_dir):
                os.mkdir(self.tbx_root_dir)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            train_foldername = "train-%s-%s" % (platform.node(), timestr)
            valid_foldername = "valid-%s-%s" % (platform.node(), timestr)
            test_foldername = "test-%s-%s" % (platform.node(), timestr)
            foldernames = [train_foldername, valid_foldername, test_foldername]
            for idx, (loader, foldername) in enumerate(zip(loaders, foldernames)):
                if loader:
                    tbx_dir = os.path.join(self.tbx_root_dir, foldername)
                    if os.path.exists(tbx_dir):
                        raise AssertionError("tbx session paths should be unique")
                    os.mkdir(tbx_dir)
                    writers[idx] = SummaryWriter(log_dir=tbx_dir, comment=self.name)
            self.logger.info("using tensorboard : tensorboard --logdir %s --port <your_port>" % self.tbx_root_dir)
        self.train_writer, self.valid_writer, self.test_writer = writers
        self.outputs = {}
        self.model = model
        self.loss = loss
        self.train_metrics = deepcopy(metrics)
        self.valid_metrics = deepcopy(metrics)
        self.test_metrics = deepcopy(metrics)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.default_dev = "cpu"
        if torch.cuda.is_available():
            self.default_dev = "cuda:0"
        self.train_dev = str(config["train_device"]) if "train_device" in config else self.default_dev
        self.valid_dev = str(config["valid_device"]) if "valid_device" in config else self.default_dev
        self.test_dev = str(config["test_device"]) if "test_device" in config else self.default_dev
        if not torch.cuda.is_available() and ("cuda" in self.train_dev or "cuda" in self.valid_dev):
            raise AssertionError("cuda not available (according to pytorch), cannot use gpu for training/validation")
        elif torch.cuda.is_available():
            nb_cuda_dev = torch.cuda.device_count()
            if "cuda:" in self.train_dev and int(self.train_dev.rsplit(":", 1)[1]) >= nb_cuda_dev:
                raise AssertionError("cuda device '%s' not currently available" % self.train_dev)
            if "cuda:" in self.valid_dev and int(self.valid_dev.rsplit(":", 1)[1]) >= nb_cuda_dev:
                raise AssertionError("cuda device '%s' not currently available" % self.valid_dev)
            if "cuda:" in self.test_dev and int(self.test_dev.rsplit(":", 1)[1]) >= nb_cuda_dev:
                raise AssertionError("cuda device '%s' not currently available" % self.test_dev)
        if "monitor" not in config or not config["monitor"]:
            raise AssertionError("missing 'monitor' field for trainer config")
        self.monitor = config["monitor"]
        if self.monitor not in self.train_metrics:
            raise AssertionError("monitored metric with name '%s' should be declared in config 'metrics' field" % self.monitor)
        self.monitor_goal = self.train_metrics[self.monitor].goal()
        self.monitor_best = None
        if self.monitor_goal == thelper.optim.Metric.minimize:
            self.monitor_best = thelper.optim.Metric.maximize
        elif self.monitor_goal == thelper.optim.Metric.maximize:
            self.monitor_best = thelper.optim.Metric.minimize
        else:
            raise AssertionError("monitored metric does not return proper optimization goal")
        train_logger_path = os.path.join(self.save_dir, "logs", "train.log")
        train_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        train_logger_fh = logging.FileHandler(train_logger_path)
        train_logger_fh.setFormatter(train_logger_format)
        self.logger.addHandler(train_logger_fh)
        self.logger.info("created training log for session '%s'" % session_name)
        self.current_lr = self._get_lr()  # for debug/display purposes only
        self.current_iter = 0
        self.start_epoch = 1

    def train(self):
        if self.train_loader:
            for epoch in range(self.start_epoch, self.epochs + 1):
                if self.scheduler:
                    self.scheduler.step(epoch=(epoch - 1))  # epoch idx is 1-based, scheduler expects 0-based
                    self.current_lr = self.scheduler.get_lr()[0]
                    self.logger.info("update learning rate to %.8f" % self.current_lr)
                self.model = self.model.to(self.train_dev)
                result = self._train_epoch(epoch, self.train_loader)
                monitor_type_key = "train/metrics"
                if self.valid_loader:
                    self.model = self.model.to(self.valid_dev)
                    result_valid = self._eval_epoch(epoch, self.valid_loader, "valid")
                    result = {**result, **result_valid}
                    monitor_type_key = "valid/metrics"
                new_best = False

                losses = {}
                monitor_vals = {}

                for key, value in result.items():
                    if key == "train/metrics":
                        if self.monitor not in value:
                            raise AssertionError("not monitoring required variable '%s' in training metrics" % self.monitor)
                        monitor_vals["train"] = value[self.monitor]
                    elif key == "valid/metrics":
                        if self.monitor not in value:
                            raise AssertionError("not monitoring required variable '%s' in validation metrics" % self.monitor)
                        monitor_vals["valid"] = value[self.monitor]
                    if (key == monitor_type_key and
                        ((self.monitor_goal == thelper.optim.Metric.minimize and value[self.monitor] < self.monitor_best) or
                         (self.monitor_goal == thelper.optim.Metric.maximize and value[self.monitor] > self.monitor_best))):
                        self.monitor_best = value[self.monitor]
                        new_best = True
                    if key == "train/loss":
                        losses["train"] = value
                    elif key == "valid/loss":
                        losses["valid"] = value
                    if not isinstance(value, dict):
                        self.logger.debug(" epoch {} result =>  {}: {}".format(epoch, str(key), value))
                    else:
                        for subkey, subvalue in value.items():
                            self.logger.debug(" epoch {} result =>  {}:{}: {}".format(epoch, str(key), str(subkey), subvalue))
                if not monitor_vals or not losses:
                    raise AssertionError("training/validation did not produce required losses & monitoring variable '%s'" % self.monitor)
                self.outputs[epoch] = result
                if new_best or (epoch % self.save_freq) == 0:
                    self._save(epoch, save_best=new_best)
            self.logger.info("training done")
            if self.test_loader:
                self.model = self.model.to(self.test_dev)
                result_test = self._eval_epoch(self.epochs, self.test_loader, "test")
                self.outputs[self.epochs] = {**self.outputs[self.epochs], **result_test}
                for key, value in self.outputs[self.epochs].items():
                    if not isinstance(value, dict):
                        self.logger.debug(" final result =>  {}: {}".format(str(key), value))
                    else:
                        for subkey, subvalue in value.items():
                            self.logger.debug(" final result =>  {}:{}: {}".format(str(key), str(subkey), subvalue))
                self.logger.info("evaluation done")
            return
        result = {}
        if self.valid_loader:
            self.model = self.model.to(self.valid_dev)
            result_valid = self._eval_epoch(self.start_epoch, self.valid_loader, "valid")
            result = {**result, **result_valid}
        if self.test_loader:
            self.model = self.model.to(self.test_dev)
            result_test = self._eval_epoch(self.start_epoch, self.test_loader, "test")
            result = {**result, **result_test}
        for key, value in result.items():
            if not isinstance(value, dict):
                self.logger.debug(" final result =>  {}: {}".format(str(key), value))
            else:
                for subkey, subvalue in value.items():
                    self.logger.debug(" final result =>  {}:{}: {}".format(str(key), str(subkey), subvalue))
        self.outputs[self.start_epoch] = result
        self.logger.info("evaluation done")
        # not saving final eval results anywhere...? todo

    @abstractmethod
    def _train_epoch(self, epoch, loader):
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, epoch, loader, eval_type="valid"):
        raise NotImplementedError

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            if "lr" in param_group:
                return param_group["lr"]

    def _save(self, epoch, save_best=False):
        fullconfig = None
        if self.config_backup_path and os.path.exists(self.config_backup_path):
            with open(self.config_backup_path, "r") as fd:
                fullconfig = json.load(fd)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        curr_state = {
            "name": self.name,
            "epoch": epoch,
            "iter": self.current_iter,
            "time": timestr,
            "host": platform.node(),
            "outputs": self.outputs[epoch],
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.monitor_best,
            "config": fullconfig
        }
        filename = "ckpt.%04d.%s.%s.pth" % (epoch, platform.node(), timestr)
        filename = os.path.join(self.checkpoint_dir, filename)
        torch.save(curr_state, filename)
        if save_best:
            filename_best = os.path.join(self.checkpoint_dir, "ckpt.best.pth")
            torch.save(curr_state, filename_best)
            self.logger.info("saving new best checkpoint @ epoch %d" % epoch)
        else:
            self.logger.info("saving checkpoint @ epoch %d" % epoch)


class ImageClassifTrainer(Trainer):

    def __init__(self, session_name, save_dir, metapack, loaders, config):
        super().__init__(session_name, save_dir, metapack, loaders, config)
        if not isinstance(self.model.task, thelper.tasks.Classification):
            raise AssertionError("expected task to be classification")
        input_keys = self.model.task.get_input_key()
        if isinstance(input_keys, str):
            self.input_keys = [input_keys]
        elif not isinstance(input_keys, list):
            raise AssertionError("input keys must be provided as a list of string")
        else:
            self.input_keys = input_keys
        label_keys = self.model.task.get_gt_key()
        if isinstance(label_keys, str):
            self.label_keys = [label_keys]
        elif not isinstance(label_keys, list):
            raise AssertionError("input keys must be provided as a list of string")
        else:
            self.label_keys = label_keys
        self.class_map = self.model.task.get_class_map()

    def _to_tensor(self, sample, dev):
        if not isinstance(sample, dict):
            raise AssertionError("trainer expects samples to come in dicts for key-based usage")
        input, label = None, None
        for key in self.input_keys:
            if key in sample:
                input = sample[key]
                break
        for key in self.label_keys:
            if key in sample:
                label = sample[key]
                break
        if input is None or label is None:
            raise AssertionError("could not find input or label keys in sample dict")
        input, label = torch.FloatTensor(input), torch.LongTensor(label)
        input, label = input.to(dev), label.to(dev)
        return input, label

    def _train_epoch(self, epoch, loader):
        if not loader:
            raise AssertionError("no available data to load")
        result = {}
        self.model.train()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.train_dev)
        total_loss = 0
        epoch_size = len(loader)
        for metric in self.train_metrics.values():
            if hasattr(metric, "set_class_map") and callable(metric.set_class_map):
                metric.set_class_map(self.class_map)
            if hasattr(metric, "set_max_accum") and callable(metric.set_max_accum):
                metric.set_max_accum(epoch_size)
            if metric.needs_reset():
                metric.reset()
        for idx, sample in enumerate(loader):
            input, label = self._to_tensor(sample, self.train_dev)
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = self.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.current_iter += 1
            for metric in self.train_metrics.values():
                metric.accumulate(pred.cpu(), label.cpu())
            self.logger.info(
                "train epoch: {}   iter: {}   batch: {}/{} ({:.0f}%)   loss: {:.6f}   {}: {:.2f}".format(
                    epoch,
                    self.current_iter,
                    idx,
                    epoch_size,
                    (idx / epoch_size) * 100.0,
                    loss.item(),
                    self.train_metrics[self.monitor].name,
                    self.train_metrics[self.monitor].eval()
                )
            )
            if self.use_tbx:
                self.train_writer.add_scalar("iter/loss", loss.item(), self.current_iter)
                self.train_writer.add_scalar("iter/lr", self.current_lr, self.current_iter)
                for metric_name, metric in self.train_metrics.items():
                    if metric.is_scalar():
                        self.train_writer.add_scalar("iter/%s" % metric_name, metric.eval(), self.current_iter)
        metric_vals = {}
        for metric_name, metric in self.train_metrics.items():
            metric_vals[metric_name] = metric.eval()
        result["train/loss"] = total_loss / epoch_size
        result["train/metrics"] = metric_vals
        if self.use_tbx:
            self.train_writer.add_scalar("epoch/loss", total_loss / epoch_size, epoch)
            self.train_writer.add_scalar("epoch/lr", self.current_lr, epoch)
            for metric_name, metric in self.train_metrics.items():
                if metric.is_scalar():
                    self.train_writer.add_scalar("epoch/%s" % metric_name, metric.eval(), epoch)
                elif hasattr(metric, "get_tbx_image") and callable(metric.get_tbx_image):
                    self.train_writer.add_image("epoch/%s" % metric_name, metric.get_tbx_image(), epoch)
        return result

    def _eval_epoch(self, epoch, loader, eval_type="valid"):
        if not loader:
            raise AssertionError("no available data to load")
        if eval_type != "valid" and eval_type != "test":
            raise AssertionError("unexpected eval type")
        result = {}
        self.model.eval()
        if eval_type == "valid":
            dev = self.valid_dev
            writer = self.valid_writer
            metrics = self.valid_metrics
        else:
            dev = self.test_dev
            writer = self.test_writer
            metrics = self.test_metrics
        with torch.no_grad():
            total_loss = 0
            for metric in metrics.values():
                if hasattr(metric, "set_class_map") and callable(metric.set_class_map):
                    metric.set_class_map(self.class_map)
                metric.reset()  # force reset here, we always evaluate from a clean state
            epoch_size = len(loader)
            for idx, sample in enumerate(loader):
                input, label = self._to_tensor(sample, dev)
                pred = self.model(input)
                loss = self.loss(pred, label)
                total_loss += loss.item()
                for metric in metrics.values():
                    metric.accumulate(pred.cpu(), label.cpu())
                # set logger to output based on timer?
                self.logger.info(
                    "{} epoch: {}   batch: {}/{} ({:.0f}%)   loss: {:.6f}   {}: {:.2f}".format(
                        eval_type,
                        epoch,
                        idx,
                        epoch_size,
                        (idx / epoch_size) * 100.0,
                        loss.item(),
                        metrics[self.monitor].name,
                        metrics[self.monitor].eval()
                    )
                )
            metric_vals = {}
            for metric_name, metric in metrics.items():
                metric_vals[metric_name] = metric.eval()
            result[eval_type + "/loss"] = total_loss / epoch_size
            result[eval_type + "/metrics"] = metric_vals
            if self.use_tbx:
                if self.current_iter > 0:
                    writer.add_scalar("iter/loss", total_loss / epoch_size, self.current_iter)
                writer.add_scalar("epoch/loss", total_loss / epoch_size, epoch)
                for metric_name, metric in metrics.items():
                    if metric.is_scalar():
                        if self.current_iter > 0:
                            writer.add_scalar("iter/%s" % metric_name, metric.eval(), self.current_iter)
                        writer.add_scalar("epoch/%s" % metric_name, metric.eval(), epoch)
                    elif hasattr(metric, "get_tbx_image") and callable(metric.get_tbx_image):
                        writer.add_image("epoch/%s" % metric_name, metric.get_tbx_image(), epoch)
        return result

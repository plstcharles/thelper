import json
import logging
import os
import sys
import time
from abc import abstractmethod

import torch
import torch.optim

import thelper.utils

logger = logging.getLogger(__name__)


def load_trainer(session_name,model,loss,metrics,optimizer,
                 scheduler,schedstep,loaders,config,resume=None):
    if "trainer" not in config or not config["trainer"]:
        raise AssertionError("config missing 'trainer' field")
    trainer_config = config["trainer"]
    if "save_dir" not in trainer_config or not trainer_config["save_dir"]:
        raise AssertionError("trainer config missing 'save_dir' field")
    save_dir = str(trainer_config["save_dir"])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir,session_name)
    overwrite = False
    old_session_name = session_name
    time.sleep(0.5)  # to make sure all debug/info prints are done, and we see the question
    while os.path.exists(save_dir) and not overwrite:
        overwrite = thelper.utils.query_yes_no("Training session at '%s' already exists; overwrite?"%save_dir)
        if not overwrite:
            session_name = thelper.utils.query_string("Please provide a new session name (old=%s):"%old_session_name)
            save_dir = os.path.join(save_dir,session_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if "type" not in trainer_config or not trainer_config["type"]:
        raise AssertionError("trainer config missing 'type' field")
    trainer_type = thelper.utils.import_class(trainer_config["type"])
    if "params" not in trainer_config:
        raise AssertionError("trainer config missing 'params' field")
    params = thelper.utils.keyvals2dict(trainer_config["params"])
    metapack = (model,loss,metrics,optimizer,scheduler,schedstep)
    trainer = trainer_type(session_name,save_dir,metapack,loaders,trainer_config,**params)
    if resume is not None:
        config_backup = json.load(open(trainer.config_path,"r"))
        if config_backup!=config:
            answer = thelper.utils.query_yes_no("Current config differs from config loaded at '%s'; continue?"%trainer.config_path)
            if answer:
                logger.warning("config mismatch with previous run; will reload anyway")
            else:
                logger.error("config mismatch with the one loaded at '%s', user aborted"%trainer.config_path)
                sys.exit(1)
        logger.info("loading checkpoint '%s'..."%resume)
        checkpoint = torch.load(resume)
        trainer.start_epoch = checkpoint["epoch"]+1
        trainer.monitor_best = checkpoint["monitor_best"]
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.outputs = checkpoint["outputs"]
        if "cuda" in trainer.train_dev:
            for state in trainer.optimizer.state.values():
                for k,v in state.items():
                    if isinstance(v,torch.Tensor):
                        state[k] = v.cuda(trainer.train_dev)
        logger.info("loaded training session '%s' @ epoch %d"%(trainer.name,trainer.start_epoch))
    else:
        json.dump(config,open(trainer.config_path,"w"),indent=4,sort_keys=False)
    return trainer


class Trainer:
    def __init__(self,session_name,save_dir,metapack,loaders,config):
        model,loss,metrics,optimizer,scheduler,schedstep = metapack
        if not model or not loss or not metrics or not optimizer or not config:
            raise AssertionError("missing input args")
        train_loader,valid_loader,test_loader = loaders
        if not (train_loader or valid_loader or test_loader):
            raise AssertionError("must provide at least one loader with available data")
        self.logger = thelper.utils.get_class_logger()
        self.name = session_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        if "epochs" not in config or not config["epochs"] or int(config["epochs"])<=0:
            raise AssertionError("bad trainer config epoch count")
        if train_loader:
            self.epochs = int(config["epochs"])
        else:
            self.epochs = 1
            self.logger.info("no training data provided, will run a single epoch on valid/test data")
        self.save_freq = int(config["save_freq"]) if "save_freq" in config else 1
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(self.save_dir,"checkpoints")
        self.config_path = os.path.join(self.save_dir,"config.json")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.outputs = {}
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedstep = schedstep
        self.config = config
        self.default_dev = "cpu"
        if torch.cuda.is_available():
            self.default_dev = "cuda:0"
        self.train_dev = str(config["train_device"]) if "train_device" in config else self.default_dev
        self.valid_dev = str(config["valid_device"]) if "valid_device" in config else self.default_dev
        if not torch.cuda.is_available() and ("cuda" in self.train_dev or "cuda" in self.valid_dev):
            raise AssertionError("cuda not available (according to pytorch), cannot use gpu for training/validation")
        elif torch.cuda.is_available():
            nb_cuda_dev = torch.cuda.device_count()
            if "cuda:" in self.train_dev and int(self.train_dev.rsplit(":",1)[1])>=nb_cuda_dev:
                raise AssertionError("cuda device '%s' not currently available"%self.train_dev)
            if "cuda:" in self.valid_dev and int(self.valid_dev.rsplit(":",1)[1])>=nb_cuda_dev:
                raise AssertionError("cuda device '%s' not currently available"%self.valid_dev)
            self.model = self.model.to(self.train_dev)
        if "monitor" not in config or not config["monitor"]:
            raise AssertionError("missing 'monitor' field for trainer config")
        self.monitor = config["monitor"]
        if self.monitor not in self.metrics:
            raise AssertionError("monitored metric should be declared in 'metrics' field")
        self.monitor_goal = self.metrics[self.monitor].goal()
        self.monitor_best = None
        if self.monitor_goal==thelper.optim.Metric.minimize:
            self.monitor_best = thelper.optim.Metric.maximize
        elif self.monitor_goal==thelper.optim.Metric.maximize:
            self.monitor_best = thelper.optim.Metric.minimize
        else:
            raise AssertionError("monitored metric does not return proper optimization goal")
        self.start_epoch = 1
        train_logger_path = os.path.join(self.save_dir,"train.log")
        train_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        train_logger_fh = logging.FileHandler(train_logger_path)
        train_logger_fh.setFormatter(train_logger_format)
        self.logger.addHandler(train_logger_fh)
        self.logger.info("created training log for session '%s'"%session_name)

    def train(self):
        if self.train_loader:
            for epoch in range(self.start_epoch,self.epochs+1):
                result = self._train_epoch(epoch,self.train_loader)
                monitor_type_key = "train_metrics"
                if self.valid_loader:
                    result_valid = self._eval_epoch(epoch,self.valid_loader,"valid")
                    result = {**result,**result_valid}
                    monitor_type_key = "valid_metrics"
                monitor_val = None
                new_best = False
                for key,value in result.items():
                    if key==monitor_type_key:
                        if self.monitor not in value:
                            raise AssertionError("not monitoring required variable '%s' in metrics"%self.monitor)
                        monitor_val = value[self.monitor]
                        if ((self.monitor_goal==thelper.optim.Metric.minimize and monitor_val<self.monitor_best) or
                                (self.monitor_goal==thelper.optim.Metric.maximize and monitor_val>self.monitor_best)):
                            self.monitor_best = monitor_val
                            new_best = True
                    self.logger.debug(' epoch result =>  {:15s}: {}'.format(str(key),value))
                if monitor_val==None:
                    raise AssertionError("training/validation did not produce required monitoring variable '%s' in metrics"%self.monitor)
                self.outputs[epoch] = result
                if new_best or (epoch%self.save_freq)==0:
                    self._save(epoch,save_best=new_best)
                if self.scheduler and (epoch%self.schedstep)==0:
                    self.scheduler.step(epoch)
                    lr = self.scheduler.get_lr()[0]
                    self.logger.info("update learning rate to %.7f"%lr)
            self.logger.info("training done")
            if self.test_loader:
                result_test = self._eval_epoch(self.epochs,self.test_loader,"test")
                self.outputs[self.epochs] = {**self.outputs[self.epochs],**result_test}
                for key,value in self.outputs[self.epochs].items():
                    self.logger.debug(' final result =>  {:15s}: {}'.format(str(key),value))
                self.logger.info("evaluation done")
            return
        result = {}
        if self.valid_loader:
            result_valid = self._eval_epoch(0,self.valid_loader,"valid")
            result = {**result,**result_valid}
        if self.test_loader:
            result_test = self._eval_epoch(0,self.test_loader,"test")
            result = {**result,**result_test}
        for key,value in result.items():
            self.logger.debug(' final result =>  {:15s}: {}'.format(str(key),value))
        self.outputs[0] = result
        self.logger.info("evaluation done")
        # not saving final eval results anywhere...? todo


    @abstractmethod
    def _train_epoch(self,epoch,loader):
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self,epoch,loader,eval_type="valid"):
        raise NotImplementedError

    def _save(self,epoch,save_best=False):
        curr_state = {
            "name":self.name,
            "epoch":epoch,
            "outputs":self.outputs[epoch],
            "state_dict":self.model.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "monitor_best":self.monitor_best,
            "config":json.load(open(self.config_path,"r"))
        }
        latest_loss = self.outputs[epoch]["train_loss"]
        filename = os.path.join(self.checkpoint_dir,"ckpt.%04d.L%.3f.tar"%(epoch,latest_loss))
        torch.save(curr_state,filename)
        if save_best:
            filename_best = os.path.join(self.checkpoint_dir,"ckpt.best.tar")
            torch.save(curr_state,filename_best)
            self.logger.info("saving new best checkpoint @ epoch %d"%epoch)
        else:
            self.logger.info("saving checkpoint @ epoch %d"%epoch)


class ImageClassifTrainer(Trainer):
    def __init__(self,session_name,save_dir,metapack,loaders,config,input_keys="0",label_keys="1"):
        super().__init__(session_name,save_dir,metapack,loaders,config)
        if isinstance(input_keys,str):
            self.input_keys = [input_keys]
        elif not isinstance(input_keys,list):
            raise AssertionError("input keys must be provided as a list of string")
        else:
            self.input_keys = input_keys
        if isinstance(label_keys,str):
            self.label_keys = [label_keys]
        elif not isinstance(label_keys,list):
            raise AssertionError("input keys must be provided as a list of string")
        else:
            self.label_keys = label_keys

    def _to_tensor(self,sample):
        if not isinstance(sample,dict):
            raise AssertionError("trainer expects samples to come in dicts for key-based usage")
        input,label = None,None
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
        input,label = torch.FloatTensor(input),torch.LongTensor(label)
        input,label = input.to(self.train_dev),label.to(self.train_dev)  # valid device still ignored, todo
        return input,label

    def _train_epoch(self,epoch,loader):
        if not loader:
            raise AssertionError("no available data to load")
        result = {}
        self.model.train()
        total_loss = 0
        for metric in self.metrics.values():
            metric.reset()
        for idx,sample in enumerate(loader):
            input,label = self._to_tensor(sample)
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = self.loss(pred,label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            for metric in self.metrics.values():
                metric.accumulate(pred.cpu(),label.cpu())
            self.logger.info(
                "train epoch: {}   batch: {}/{} ({:.0f}%)   loss: {:.6f}   {}: {:.2f}".format(
                    epoch,
                    idx,
                    len(loader),
                    (idx/len(loader))*100.0,
                    loss.item(),
                    self.metrics[self.monitor].name,
                    self.metrics[self.monitor].eval()
                )
            )
        metric_vals = {}
        for metric_name,metric in self.metrics.items():
            metric_vals[metric_name] = metric.eval()
        result["train_loss"] = total_loss/len(loader)
        result["train_metrics"] = metric_vals
        return result

    def _eval_epoch(self,epoch,loader,eval_type="valid"):
        if not loader:
            raise AssertionError("no available data to load")
        result = {}
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for metric in self.metrics.values():
                metric.reset()
            for idx,sample in enumerate(loader):
                input,label = self._to_tensor(sample)
                pred = self.model(input)
                loss = self.loss(pred,label)
                total_loss += loss.item()
                for metric in self.metrics.values():
                    metric.accumulate(pred.cpu(),label.cpu())
                # set logger to output based on timer?
                self.logger.info(
                    "{} epoch: {}   batch: {}/{} ({:.0f}%)   loss: {:.6f}   {}: {:.2f}".format(
                        eval_type,
                        epoch,
                        idx,
                        len(loader),
                        (idx/len(loader))*100.0,
                        loss.item(),
                        self.metrics[self.monitor].name,
                        self.metrics[self.monitor].eval()
                    )
                )
            metric_vals = {}
            for metric_name,metric in self.metrics.items():
                metric_vals[metric_name] = metric.eval()
            result[eval_type+"_loss"] = total_loss/len(loader)
            result[eval_type+"_metrics"] = metric_vals
        return result

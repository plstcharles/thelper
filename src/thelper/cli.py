"""
Command-line module, for use with __main__ entrypoint or external apps.
"""

import argparse
import json
import logging
import os
import sys
import time
from copy import copy

import torch

import thelper

logging.basicConfig(level=logging.INFO)


def load_datasets(config,data_root):
    logger = thelper.utils.get_func_logger()
    logger.debug("loading datasets config")
    if "datasets" not in config or not config["datasets"]:
        raise AssertionError("config missing 'datasets' field (can be dict or str)")
    datasets_config = config["datasets"]
    if isinstance(datasets_config,str):
        if os.path.isfile(datasets_config) and os.path.splitext(datasets_config)[1]==".json":
            datasets_config = json.load(open(datasets_config))
        else:
            raise AssertionError("'datasets' string should point to valid json file")
    logger.debug("loading datasets templates")
    if not isinstance(datasets_config,dict):
        raise AssertionError("invalid datasets config type")
    dataset_templates,task = thelper.data.load_dataset_templates(datasets_config,data_root)
    logger.debug("loading data usage config")
    if "data_config" not in config or not config["data_config"]:
        raise AssertionError("config missing 'data_config' field")
    data_config = thelper.data.DataConfig(config["data_config"])
    # if hasattr(data_config,"summary"):
    #     data_config.summary()
    logger.debug("splitting datasets and creating loaders")
    train_loader,valid_loader,test_loader = data_config.get_data_split(dataset_templates)
    return task,train_loader,valid_loader,test_loader


def load_model(config,task):
    logger = thelper.utils.get_func_logger()
    logger.debug("loading model")
    if "model" not in config or not config["model"]:
        raise AssertionError("config missing 'model' field")
    model = thelper.modules.load_model(config["model"],task)
    if hasattr(model,"summary"):
        model.summary()
    return model


def load_train_cfg(config,model):
    logger = thelper.utils.get_func_logger()
    logger.debug("loading loss & metrics configurations")
    if "loss" not in config or not config["loss"]:
        raise AssertionError("config missing 'loss' field")
    loss = thelper.optim.load_loss(config["loss"])
    if hasattr(loss,"summary"):
        loss.summary()
    if "metrics" not in config or not config["metrics"]:
        raise AssertionError("config missing 'metrics' field")
    metrics = thelper.optim.load_metrics(config["metrics"])
    for metric_name,metric in metrics.items():
        if hasattr(metric,"summary"):
            logger.info("parsed metric category '%s'"%metric_name)
            metric.summary()
    logger.debug("loading optimization & scheduler configurations")
    if "optimization" not in config or not config["optimization"]:
        raise AssertionError("config missing 'optimization' field")
    optimizer,scheduler,schedstep = thelper.optim.load_optimization(model,config["optimization"])
    return loss,metrics,optimizer,scheduler,schedstep


def get_save_dir(out_root,session_name,config,resume=False):
    logger = thelper.utils.get_func_logger()
    save_dir = out_root
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir,session_name)
    if not resume:
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
        config_backup_path = os.path.join(save_dir,"config.json")
        json.dump(config,open(config_backup_path,"w"),indent=4,sort_keys=False)
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        config_backup_path = os.path.join(save_dir,"config.json")
        if os.path.exists(config_backup_path):
            config_backup = json.load(open(config_backup_path,"r"))
            if config_backup!=config:
                answer = thelper.utils.query_yes_no("Config backup in '%s' differs from config loaded through checkpoint; overwrite?"%config_backup_path)
                if answer:
                    logger.warning("config mismatch with previous run; will overwrite backup in save directory")
                else:
                    logger.error("config mismatch with previous run; user aborted")
                    sys.exit(1)
        json.dump(config,open(config_backup_path,"w"),indent=4,sort_keys=False)
    return save_dir


def create_session(config,data_root,save_dir,display_graphs=False):
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    save_dir = get_save_dir(save_dir,session_name,config)
    logger.info("Creating new training session '%s'..."%session_name)
    task,train_loader,valid_loader,test_loader = load_datasets(config,data_root)
    if display_graphs and logger.isEnabledFor(logging.DEBUG):
        if not train_loader:
            raise AssertionError("cannot draw sample example graph, train loader is empty")
        train_loader_copy = copy(train_loader)
        data_iter = iter(train_loader_copy)
        # noinspection PyUnresolvedReferences
        data_sample = data_iter.next()
        thelper.utils.draw_sample(data_sample,block=True)
    model = load_model(config,task)
    loss,metrics,optimizer,scheduler,schedstep = load_train_cfg(config,model)
    loaders = (train_loader,valid_loader,test_loader)
    trainer = thelper.train.load_trainer(session_name,save_dir,config,model,loss,
                                         metrics,optimizer,scheduler,schedstep,loaders)
    logger.debug("starting trainer")
    trainer.train()
    logger.debug("all done")


def resume_session(ckptdata,data_root,save_dir,map_location=None,display_graphs=False):
    logger = thelper.utils.get_func_logger()
    if "config" not in ckptdata or not ckptdata["config"]:
        raise AssertionError("checkpoint data missing 'config' field")
    config = ckptdata["config"]
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    save_dir = get_save_dir(save_dir,session_name,config,resume=True)
    logger.info("loading training session '%s' objects..."%session_name)
    task,train_loader,valid_loader,test_loader = load_datasets(config,data_root)
    if display_graphs and logger.isEnabledFor(logging.DEBUG):
        if not train_loader:
            raise AssertionError("cannot draw sample example graph, train loader is empty")
        train_loader_copy = copy(train_loader)
        data_iter = iter(train_loader_copy)
        # noinspection PyUnresolvedReferences
        data_sample = data_iter.next()
        thelper.utils.draw_sample(data_sample,block=True)
    model = load_model(config,task)
    model.load_state_dict(ckptdata["state_dict"])
    loss,metrics,optimizer,scheduler,schedstep = load_train_cfg(config,model)
    optimizer.load_state_dict(ckptdata["optimizer"])
    loaders = (train_loader,valid_loader,test_loader)
    trainer = thelper.train.load_trainer(session_name,save_dir,config,model,loss,
                                         metrics,optimizer,scheduler,schedstep,loaders)
    trainer.start_epoch = ckptdata["epoch"]+1
    trainer.monitor_best = ckptdata["monitor_best"]
    trainer.outputs = ckptdata["outputs"]
    logger.info("resuming training session '%s' @ epoch %d"%(trainer.name,trainer.start_epoch))
    trainer.train()
    logger.debug("all done")


def main(args=None):
    ap = argparse.ArgumentParser(description='thelper model trainer application')
    ap.add_argument("--version",default=False,action="store_true",help="prints the version of the library and exits")
    ap.add_argument("-l","--log",default="thelper.log",type=str,help="path to the output log file (default: './thelper.log')")
    ap.add_argument("-v","--verbose",action="count",default=0,help="set logging terminal verbosity level (additive)")
    ap.add_argument("-g","--display-graphs",action="store_true",default=False,help="toggles whether graphs should be displayed or not")
    ap.add_argument("-d","--data-root",default="./data/",type=str,help="path to the root directory passed to dataset interfaces for parsing")
    subparsers = ap.add_subparsers(title="Operating mode")
    new_session_ap = subparsers.add_parser("new",help="creates a new session from a config file")
    new_session_ap.add_argument("cfg_path",type=str,help="path to the training configuration file")
    new_session_ap.add_argument("save_dir",type=str,help="path to the root directory where checkpoints should be saved")
    new_session_ap.set_defaults(new_session=True)
    resume_session_ap = subparsers.add_parser("resume",help="resume a session from a checkpoint file")
    resume_session_ap.add_argument("ckpt_path",type=str,help="path to the checkpoint to resume training from")
    resume_session_ap.add_argument("save_dir",type=str,help="path to the root directory where checkpoints should be saved")
    resume_session_ap.add_argument("-m","--map-location",default=None,help="map location for loading data (default=None)")
    resume_session_ap.set_defaults(new_session=False)
    args = ap.parse_args(args=args)
    if args.verbose>2:
        log_level = logging.NOTSET
    elif args.verbose>1:
        log_level = logging.DEBUG
    elif args.verbose==1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    if args.version:
        print(thelper.__version__)
        return
    logging.getLogger().setLevel(logging.NOTSET)
    thelper.logger.propagate = 0
    logger_format = logging.Formatter("[%(asctime)s - %(name)s - %(process)s:%(thread)s] %(levelname)s : %(message)s")
    if args.log:
        logger_fh = logging.FileHandler(args.log)
        logger_fh.setLevel(logging.DEBUG)
        logger_fh.setFormatter(logger_format)
        thelper.logger.addHandler(logger_fh)
    logger_ch = logging.StreamHandler()
    logger_ch.setLevel(log_level)
    logger_ch.setFormatter(logger_format)
    thelper.logger.addHandler(logger_ch)
    thelper.logger.debug("checking dataset root '%s'..."%args.data_root)
    if not os.path.exists(args.data_root) or not os.path.isdir(args.data_root):
        raise AssertionError("invalid data root folder at '%s'; please specify the correct path via --data-root=PATH")
    if args.new_session:
        thelper.logger.debug("parsing config at '%s'"%args.cfg_path)
        config = json.load(open(args.cfg_path))
        create_session(config,args.data_root,args.save_dir,display_graphs=args.display_graphs)
    else:
        thelper.logger.debug("parsing checkpoint at '%s'"%args.ckpt_path)
        ckptdata = torch.load(args.ckpt_path,map_location=args.map_location)
        resume_session(ckptdata,args.data_root,args.save_dir,map_location=args.map_location,display_graphs=args.display_graphs)

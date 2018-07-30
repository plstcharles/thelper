"""
Command-line module, for use with __main__ entrypoint or external apps.
"""

import argparse
import json
import logging
import os
from copy import copy

import torch

import thelper

logging.basicConfig(level=logging.INFO)


def train(config,resume,data_root,display_graphs=False):
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("Instantiating training session '%s'..."%session_name)
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
    dataset_templates = thelper.data.load_dataset_templates(datasets_config,data_root)
    logger.debug("loading data usage config")
    if "data_config" not in config or not config["data_config"]:
        raise AssertionError("config missing 'data_config' field")
    data_config = thelper.data.DataConfig(config["data_config"])
    # if hasattr(data_config,"summary"):
    #     data_config.summary()
    logger.debug("splitting datasets and creating loaders")
    train_loader,valid_loader,test_loader = data_config.get_data_split(dataset_templates)
    if display_graphs and logger.isEnabledFor(logging.DEBUG):
        train_loader_copy = copy(train_loader)
        data_iter = iter(train_loader_copy)
        # noinspection PyUnresolvedReferences
        data_sample = data_iter.next()
        thelper.utils.draw_sample(data_sample)
    logger.debug("loading model")
    if "model" not in config or not config["model"]:
        raise AssertionError("config missing 'model' field")
    model = thelper.modules.load_model(config["model"])
    if hasattr(model,"summary"):
        model.summary()
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
    logger.debug("loading trainer configuration")
    loaders = (train_loader,valid_loader)
    trainer = thelper.train.load_trainer(session_name,model,loss,metrics,optimizer,
                                         scheduler,schedstep,loaders,config,resume=resume)
    logger.debug("starting trainer")
    trainer.train()
    logger.debug("training done")


def main(args=None):
    ap = argparse.ArgumentParser(description='thelper model trainer application')
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-c","--config",type=str,help="path to the training configuration file")
    input_group.add_argument("-r","--resume",type=str,help="path to the checkpoint to resume training from")
    input_group.add_argument("--version",default=False,action="store_true",help="prints the version of the library and exits")
    ap.add_argument("-l","--log",default="thelper.log",type=str,help="path to the output log file (default: './train.log')")
    ap.add_argument("-v","--verbose",action="count",default=0,help="set logging terminal verbosity level (additive)")
    ap.add_argument("-g","--display-graphs",action="store_true",default=False,help="toggles whether graphs should be displayed or not")
    ap.add_argument("-d","--data-root",default="./data/",type=str,help="path to the data root directory for parsing datasets")
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
    config = None
    if args.resume is not None:
        config = torch.load(args.resume)["config"]
        if not config:
            raise AssertionError("torch checkpoint loading failed")
    elif args.config is not None:
        config = json.load(open(args.config))
        if "name" not in config or not config["name"]:
            raise AssertionError("model configuration must be named")
    thelper.logger.info("parsed config '%s' from cli entrypoint"%config["name"])
    train(config,args.resume,args.data_root,display_graphs=args.display_graphs)

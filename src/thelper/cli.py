"""
Command-line module, for use with __main__ entrypoint or external apps.
"""

import argparse
import json
import logging
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

import thelper

logging.basicConfig(level=logging.INFO)


def create_session(config, data_root, save_dir):
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.info("Creating new training session '%s'..." % session_name)
    task, train_loader, valid_loader, test_loader = thelper.data.load(config, data_root, save_dir)
    model = thelper.modules.load_model(config, task, save_dir)
    loaders = (train_loader, valid_loader, test_loader)
    trainer = thelper.train.load_trainer(session_name, save_dir, config, model, loaders)
    logger.debug("starting trainer")
    trainer.train()
    logger.debug("all done")
    return 0


def extract(config, resume, data_root):
    from tqdm import tqdm
    import pickle as pkl
    import bz2

    logger = thelper.utils.get_func_logger()

    thelper.utils.check_key("visualizer", config, 'config')
    visualizer_config = config['visualizer']

    thelper.utils.check_key("produce_features", visualizer_config, 'visualizer_config')
    produce_features = visualizer_config['produce_features']

    if not produce_features:
        logger.info('bypassing features extraction')
        return

    thelper.utils.check_key("name", config, 'config')

    session_name = config["name"]
    logger.info("Instantiating training session '%s'..."%session_name)
    logger.debug("loading datasets config")

    thelper.utils.check_key("datasets", config, 'config', "config missing 'datasets' field (can be dict or str)")
    datasets_config = config["datasets"]

    if isinstance(datasets_config,str):
        if os.path.isfile(datasets_config) and os.path.splitext(datasets_config)[1]==".json":
            datasets_config = json.load(open(datasets_config))
        else:
            raise AssertionError("'datasets' string should point to valid json file")
    logger.debug("loading datasets templates")
    if not isinstance(datasets_config,dict):
        raise AssertionError("invalid datasets config type")
    datasets,task = thelper.data.load_datasets(datasets_config,data_root)
    logger.debug("loading data usage cofrom tqdm import tqdmnfig")
    thelper.utils.check_key("data_config", config, 'config')
    data_config = thelper.data.DataConfig(config["data_config"])

    # if hasattr(data_config,"summary"):
    #     data_config.summary()
    logger.debug("splitting datasets and creating loaders")
    data_loader,valid_loader,test_loader = data_config.get_data_split(datasets,task)

    model = thelper.modules.load_model(config,task)

    save_dir=os.path.join(config['visualizer']['save_dir'], 'features/%s/features' % config['model']['type'])

    stop=False
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        stop =not thelper.utils.query_yes_no("Directory already exists. Continue?")

    if stop:
        logger.debug("quit ...")
        return

    import cv2

    logger.debug("starting extractor")

    produce_features=visualizer_config['produce_features']

    if produce_features:

        pbar = tqdm(total=len(data_loader)*data_config.batch_size)
        k=0
        for iter, data in enumerate(data_loader):
            features = model(data['image'])
            for feature, image, label_name in zip(features, data['image'], data['label_name']):
                image = image.cpu().data.numpy().transpose(1, 2, 0)
                #cv2.imshow('image', cv2.resize(image, dsize=(64,64)))
                #cv2.waitKey(10)
                prefix='%06i.pkl.bz2' % k
                feature_path = os.path.join(save_dir, prefix)
                feature_data = {'features': feature, 'image': image.transpose(2,0,1), 'label_name': label_name}
                pkl.dump(feature_data, bz2.open(feature_path, 'wb'))
                #print(prefix)
                k+=1
            pbar.update(data_config.batch_size)
        pbar.close()
    # data_features.append({'features': feature, 'metadata': data['label_name']})

    logger.debug("all done")


def compute_features_pca(config, resume, data_root):
    from tqdm import tqdm
    import pickle as pkl
    import bz2

    logger = thelper.utils.get_func_logger()

    thelper.utils.check_key("visualizer", config, 'config')
    visualizer_config = config['visualizer']

    thelper.utils.check_key("compute_pca", visualizer_config, 'visualizer_config')
    compute_pca = visualizer_config['compute_pca']

    if not compute_pca:
        logger.info('bypassing pca computation')
        return

    thelper.utils.check_key("name", config, 'config')
    session_name = config["name"]
    logger.info("Instantiating pca computation session '%s'..."%session_name)
    logger.debug("loading datasets config")

    save_dir =os.path.join(config['visualizer']['save_dir'], 'features/%s' % config['model']['type'])

    stop=False
    if not os.path.exists(save_dir):
        raise Exception('saved directory not found: %s' % save_dir)

    features_saved_dir = os.path.join(save_dir, 'features')
    if not os.path.exists(save_dir):
        raise Exception('features directory not found: %s' % features_saved_dir)

    logger.debug("starting pca computation")

    dataset_config ={}
    dataset_config['match_pattern'] = '*.pkl.bz2'
    dataset_config['batch_size'] = 64
    dataset_config['num_workers'] = 4
    dataset_config['shuffle'] = True
    dataset_config['features_datasets_directory'] = features_saved_dir

    ofdir = os.path.join(save_dir,'pca')
    if not os.path.exists(ofdir):
        os.makedirs(ofdir)

    ofn = os.path.join(ofdir, 'pca.pkl.bz2')

    n, data_loader = thelper.features.load_datasets_dataset(dataset_config)
    step_size = 1
    X=[]
    pbar = tqdm(total=n)
    for iter, (fns) in enumerate(data_loader):
        for fn in fns:
            #print("Loading features : %s" % fn)
            dataset = pkl.load(bz2.open(fn, 'rb'))
            feature = dataset['features']
            feature_size = feature.shape[0]
            X.append(feature)
            pbar.update(1)
    pbar.close()

    logger.info('create features stack')
    X = torch.stack(X)
    logger.info(X.shape)

    logger.info("computing PCA")

    X_mean = torch.mean(X, dim=0)
    X = (X - X_mean.expand_as(X))

    U, S, V = torch.svd(torch.t(X))

    logger.info("saving PCA : %s" % ofn)
    torch.save([U.cpu(), S.cpu(), X_mean.cpu()], ofn)

    # Save matplotlib graph
    basename = os.path.basename(ofn)
    s = S.cpu().data.numpy()
    sum = np.sum(s)
    s_norm = s / sum
    s_norm_cum = np.cumsum(s_norm)

    fig, ax = plt.subplots()
    ax.bar(range(feature_size), s / s[0], color='r', edgecolor='r', label='Eigen Values')
    ax.plot(range(feature_size), s_norm_cum, label='Cumulative values')

    plt.ylabel('Eigen values')
    plt.xlabel('Features')
    legend = ax.legend(loc='center right', shadow=True)
    plt.show()

    gofn = os.path.join(ofdir, '%s.png' % basename)
    fig.savefig(gofn)
    plt.close(fig)

    logger.debug("all done")


def features_visualization(config, resume, data_root):
    from tqdm import tqdm
    import pickle as pkl
    import bz2
    import cv2

    logger = thelper.utils.get_func_logger()

    thelper.utils.check_key("name", config, 'config')
    session_name = config["name"]

    logger.info("Instantiating viz  computation session '%s'..." % session_name)
    logger.debug("loading datasets config")

    thelper.utils.check_key("visualizer", config, 'config')
    visualizer_config = config['visualizer']

    thelper.utils.check_key("compute_viz", visualizer_config, 'visualizer_config')
    compute_viz = visualizer_config['compute_viz']

    if not compute_viz:
        logger.info('bypassing viz computation')
        return

    thelper.utils.check_key("apply_pca", visualizer_config, 'visualizer_config')
    apply_pca = visualizer_config['apply_pca']



    save_dir = os.path.join(config['visualizer']['save_dir'], 'features/%s' % config['model']['type'])

    stop = False
    if not os.path.exists(save_dir):
        raise Exception('features directory not found: %s' % save_dir)

    logger.debug("feature visualization")

    dataset_config = {}
    dataset_config['match_pattern'] = '*.pkl.bz2'
    dataset_config['batch_size'] = 64
    dataset_config['num_workers'] = 4
    dataset_config['shuffle'] = True
    dataset_config['features_datasets_directory'] = os.path.join(save_dir, 'features')
    dataset_config['variance_cutoff'] = 1.0

    n, data_loader = thelper.features.load_datasets_dataset(dataset_config)
    step_size = 1
    X = []

    labels=[]
    images=[]

    max_sprite_size_sq=8192*8192
    roi_size = 64
    roi_size_sq=roi_size*roi_size
    max_data = int(max_sprite_size_sq/roi_size_sq)
    logger.info('maximum number of data that can be presented: %i with size %ix%i' % (max_data, roi_size,roi_size))
    k=0
    pbar = tqdm(total=max_data)

    label_name_list = ['accept']
    for iter, (fns) in enumerate(data_loader):
        if k >= max_data:
            break
        for fn in fns:
            # print("Loading features : %s" % fn)
            dataset = pkl.load(bz2.open(fn, 'rb'))
            feature = dataset['features']
            labels.append(dataset['label_name'])
            if dataset['label_name'] in label_name_list:
                images.append(torch.from_numpy(cv2.resize(dataset['image'],dsize=(roi_size,roi_size)).transpose(2,0,1)))
                feature_size = feature.shape[0]
                X.append(feature)
            pbar.update(1)
            k+=1

    pbar.close()

    X = torch.stack(X)

    if apply_pca:

        ifdir = os.path.join(save_dir, 'pca')
        pca_file_path = os.path.join(ifdir, 'pca.pkl.bz2')

        if not os.path.exists(pca_file_path):
            logger.info('PCA file not found: %s' % pca_file_path)

        variance_cutoff = dataset_config['variance_cutoff']
        U, S, X_Mean, cut_idx = thelper.features.load_pca_data(pca_file_path, variance_cutoff=variance_cutoff)

        X = X - X_Mean.expand_as(X)
        features = torch.mm(X, U[:, :cut_idx])
    else:
        features=X

    images = torch.stack(images, dim=0)

    from tensorboardX import SummaryWriter

    log_dir = os.path.join(save_dir, 'viz')
    writer = SummaryWriter(log_dir=log_dir)

    writer.add_embedding(features, metadata=labels, label_img=images)

    writer.close()

    logger.debug("all done")



    return


def resume_session(ckptdata, data_root, save_dir, config=None, eval_only=False):
    logger = thelper.utils.get_func_logger()
    if not config:
        if "config" not in ckptdata or not ckptdata["config"]:
            raise AssertionError("checkpoint data missing 'config' field")
        config = ckptdata["config"]
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config, resume=True)
    logger.info("loading training session '%s' objects..." % session_name)
    task, train_loader, valid_loader, test_loader = thelper.data.load(config, data_root, save_dir)
    if "task" not in ckptdata:
        logger.warning("cannot verify that checkpoint task is same as current task, might cause key or class mapping issues")
    elif task != ckptdata["task"]:
        raise AssertionError("checkpoint task mismatch with current task")
    model = thelper.modules.load_model(config, task, save_dir)
    loaders = (None if eval_only else train_loader, valid_loader, test_loader)
    trainer = thelper.train.load_trainer(session_name, save_dir, config, model, loaders, ckptdata=ckptdata)
    if eval_only:
        logger.info("evaluating session '%s' checkpoint @ epoch %d" % (trainer.name, trainer.current_epoch))
        trainer.eval()
    else:
        logger.info("resuming training session '%s' @ epoch %d" % (trainer.name, trainer.current_epoch))
        trainer.train()
    logger.debug("all done")
    return 0


def main(args=None):
    ap = argparse.ArgumentParser(description='thelper model trainer application')
    ap.add_argument("--version", default=False, action="store_true", help="prints the version of the library and exits")
    ap.add_argument("-l", "--log", default="thelper.log", type=str, help="path to the top-level log file (default: 'thelper.log')")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="set logging terminal verbosity level (additive)")
    ap.add_argument("-d", "--data-root", default=None, type=str, help="path to the root directory passed to dataset interfaces for parsing")
    subparsers = ap.add_subparsers(title="Operating mode", dest="mode")
    new_session_ap = subparsers.add_parser("new", help="creates a new session from a config file")
    new_session_ap.add_argument("cfg_path", type=str, help="path to the training configuration file")
    new_session_ap.add_argument("save_dir", type=str, help="path to the root directory where checkpoints should be saved")
    cl_new_session_ap = subparsers.add_parser("cl_new", help="creates a new session from a config file for the cluster")
    cl_new_session_ap.add_argument("cfg_path", type=str, help="path to the training configuration file")
    cl_new_session_ap.add_argument("save_dir", type=str, help="path to the root directory where checkpoints should be saved")
    cl_new_session_ap.add_argument("-n", "--nb-devices", default=2, type=int, help="number of devices to test for availability on cluster")
    resume_session_ap = subparsers.add_parser("resume", help="resume a session from a checkpoint file")
    resume_session_ap.add_argument("ckpt_path", type=str, help="path to the checkpoint to resume training from")
    resume_session_ap.add_argument("-s", "--save-dir", default=None, type=str, help="path to the root directory where checkpoints should be saved")
    resume_session_ap.add_argument("-m", "--map-location", default=None, help="map location for loading data (default=None)")
    resume_session_ap.add_argument("-c", "--override-cfg", default=None, help="override config file path (default=None)")
    resume_session_ap.add_argument("-e", "--eval-only", default=False, action="store_true", help="only run evaluation pass (valid+test)")
    args = ap.parse_args(args=args)
    if args.verbose > 2:
        log_level = logging.NOTSET
    elif args.verbose > 1:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    if args.version:
        print(thelper.__version__)
        return 0
    if args.mode is None:
        ap.print_help()
        return 1
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
    if args.data_root:
        thelper.logger.debug("checking dataset root '%s'..." % args.data_root)
        if not os.path.exists(args.data_root) or not os.path.isdir(args.data_root):
            raise AssertionError("invalid data root folder at '%s'; please specify a valid path via --data-root=PATH")
    if args.mode == "new":
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        config = json.load(open(args.cfg_path))
        return create_session(config, args.data_root, args.save_dir)
    elif args.mode == "cl_new":
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        config = json.load(open(args.cfg_path))
        nb_devices = args.nb_devices
        device_id = thelper.utils.test_cuda_device(nb_devices)
        if device_id < 0 or device_id >= nb_devices:
            raise AssertionError("no cuda device available")
        config["trainer"]["trainer"]["train_device"] = "cuda:%i" % device_id
        config["trainer"]["trainer"]["valid_device"] = "cuda:%i" % device_id
        config["trainer"]["trainer"]["test_device"] = "cuda:%i" % device_id
        return create_session(config, args.data_root, args.save_dir)
    elif args.mode == "resume":
        thelper.logger.debug("parsing checkpoint at '%s'" % args.ckpt_path)
        ckptdata = torch.load(args.ckpt_path, map_location=args.map_location)
        override_config = None
        if args.override_cfg:
            thelper.logger.debug("parsing override config at '%s'" % args.override_cfg)
            override_config = json.load(open(args.override_cfg))
        save_dir = args.save_dir
        if save_dir is None:
            save_dir = os.path.abspath(os.path.join(os.path.dirname(args.ckpt_path), "../.."))
        return resume_session(ckptdata, args.data_root, save_dir, config=override_config, eval_only=args.eval_only)

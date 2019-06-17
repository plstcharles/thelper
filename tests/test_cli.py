import copy
import os
import shutil

import numpy as np
import pytest
import torch

import thelper

test_save_path = ".pytest_cache"

test_create_simple_path = os.path.join(test_save_path, "simple")
test_create_simple_images_path = os.path.join(test_save_path, "simple_images")


@pytest.fixture
def simple_config(request):
    def fin():
        shutil.rmtree(test_create_simple_path, ignore_errors=True)
        shutil.rmtree(test_create_simple_images_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(test_create_simple_images_path, exist_ok=True)
    for cls in range(10):
        os.makedirs(os.path.join(test_create_simple_images_path, str(cls)), exist_ok=True)
        for idx in range(10):
            open(os.path.join(test_create_simple_images_path, str(cls), str(idx) + ".jpg"), "a").close()
    return {
        "name": "simple",
        "bypass_queries": True,
        "datasets": {
            "dset": {
                "type": "thelper.data.ImageFolderDataset",
                "params": {
                    "root": test_create_simple_images_path
                }
            }
        },
        "loaders": {
            "shuffle": True,
            "workers": 0,
            "batch_size": 32,
            "skip_class_balancing": True,
            "train_split": {
                "dset": 0.9
            },
            "valid_split": {
                "dset": 0.1
            }
        },
        "model": {
            "type": "thelper.nn.resnet.ResNet"
        },
        "trainer": {
            "type": "thelper.train.ImageClassifTrainer",
            "epochs": 2,
            "optimization": {
                "loss": {
                    "type": "torch.nn.CrossEntropyLoss"
                },
                "optimizer": {
                    "type": "torch.optim.Adam",
                    "params": {
                        "lr": 0.001
                    }
                }
            }
        }
    }


def test_create_session_nameless(simple_config, mocker):
    fake_train = mocker.patch.object(thelper.train.base.Trainer, "train")
    fake_eval = mocker.patch.object(thelper.train.base.Trainer, "eval")
    del simple_config["name"]
    with pytest.raises(AssertionError):
        thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 0
    assert fake_eval.call_count == 0


def test_create_session_train(simple_config, mocker):
    fake_train = mocker.patch.object(thelper.train.base.Trainer, "train")
    fake_eval = mocker.patch.object(thelper.train.base.Trainer, "eval")
    thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 1
    assert fake_eval.call_count == 0
    assert os.path.isdir(test_create_simple_path)


def test_create_session_eval(simple_config, mocker):
    fake_train = mocker.patch.object(thelper.train.base.Trainer, "train")
    fake_eval = mocker.patch.object(thelper.train.base.Trainer, "eval")
    del simple_config["loaders"]["train_split"]
    thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 0
    assert fake_eval.call_count == 1
    assert os.path.isdir(test_create_simple_path)


def test_resume_session(simple_config, mocker):
    log_iter = 0

    def get_log_stamp():
        nonlocal log_iter
        log_iter += 1
        return "fake-20000101-%06d" % log_iter

    fake_logstamp = mocker.patch("thelper.utils.get_log_stamp")
    fake_logstamp.side_effect = get_log_stamp
    with pytest.raises(AssertionError):
        thelper.cli.resume_session(None, test_save_path, simple_config)
    with pytest.raises(AssertionError):
        thelper.cli.resume_session({"dummy": None}, test_save_path)
    with pytest.raises(AssertionError):
        thelper.cli.resume_session({"config": {}}, test_save_path)
    fake_train = mocker.patch.object(thelper.train.classif.ImageClassifTrainer, "train_epoch", return_value=[0, 0])
    fake_eval = mocker.patch.object(thelper.train.classif.ImageClassifTrainer, "eval_epoch")
    with pytest.raises((FileNotFoundError, AssertionError)):
        _ = thelper.utils.load_checkpoint(test_create_simple_path)
    thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 2
    assert fake_eval.call_count == 2
    ckptdata = thelper.utils.load_checkpoint(test_create_simple_path)
    ckptdata_fake = {key: val for key, val in ckptdata.items() if key != "task"}
    with pytest.raises(AssertionError):
        thelper.cli.resume_session(ckptdata_fake, test_save_path, eval_only=True)
    nameless_ckptdata = copy.deepcopy(ckptdata)
    del nameless_ckptdata["config"]["name"]
    with pytest.raises(AssertionError):
        thelper.cli.resume_session(nameless_ckptdata, test_save_path, eval_only=True)
    old_task = ckptdata["task"]
    thelper.cli.resume_session(ckptdata, test_save_path, eval_only=True)
    latest_ckptdata = thelper.utils.load_checkpoint(test_create_simple_path, always_load_latest=True)
    assert old_task == latest_ckptdata["task"]
    assert fake_train.call_count == 2
    assert fake_eval.call_count == 3
    thelper.cli.resume_session(ckptdata, test_save_path)
    latest_ckptdata = thelper.utils.load_checkpoint(test_create_simple_path, always_load_latest=True)
    assert old_task == latest_ckptdata["task"]
    assert fake_train.call_count == 3
    assert fake_eval.call_count == 4
    override_config = ckptdata["config"]
    override_config["trainer"]["epochs"] = 3
    thelper.cli.resume_session(ckptdata, test_save_path, override_config)
    latest_ckptdata = thelper.utils.load_checkpoint(test_create_simple_path, always_load_latest=True)
    assert old_task == latest_ckptdata["task"]
    assert fake_train.call_count == 5
    assert fake_eval.call_count == 6
    override_config = ckptdata["config"]
    new_finetune_task = {
        "type": "thelper.tasks.Classification",
        "params": {
            "class_names": [
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
            ],
            "input_key": "image",
            "label_key": "label"
        }
    }
    override_config["datasets"]["dset"]["task"] = new_finetune_task
    fake_query = mocker.patch("thelper.utils.query_string", return_value="new")
    thelper.cli.resume_session(ckptdata, test_save_path)
    latest_ckptdata = thelper.utils.load_checkpoint(test_create_simple_path, always_load_latest=True)
    assert str(thelper.tasks.create_task(new_finetune_task)) == latest_ckptdata["task"]
    assert fake_query.call_count == 1
    _ = mocker.patch("thelper.utils.query_string", return_value="old")
    thelper.cli.resume_session(ckptdata, test_save_path)
    latest_ckptdata = thelper.utils.load_checkpoint(test_create_simple_path, always_load_latest=True)
    assert old_task == latest_ckptdata["task"]


def test_visualize_data(simple_config, mocker):
    fake_draw = mocker.patch("thelper.utils.draw_minibatch")
    fake_imread = mocker.patch("cv2.imread", return_value=["ohhai"])
    fake_config_bad_viz = copy.deepcopy(simple_config)
    fake_config_bad_viz["viz"] = "bad"
    with pytest.raises(AssertionError):
        thelper.cli.visualize_data(fake_config_bad_viz)
    fake_config_bad_viz["viz"] = {"kwargs": "bad"}
    with pytest.raises(AssertionError):
        thelper.cli.visualize_data(fake_config_bad_viz)
    assert fake_draw.call_count == 0
    config_viz_no_loaders = copy.deepcopy(simple_config)
    del config_viz_no_loaders["loaders"]
    query_idx = 0

    def query_dset(*args, **kwargs):
        nonlocal query_idx
        if query_idx == 0:
            query_idx += 1
            return "dset"
        else:
            return "quit"
    fake_query = mocker.patch("thelper.utils.query_string")
    fake_query.side_effect = query_dset
    thelper.cli.visualize_data(config_viz_no_loaders)
    assert fake_draw.call_count == 100
    assert fake_imread.call_count == 100
    query_idx = 0

    def query_loader(*args, **kwargs):
        nonlocal query_idx
        if query_idx == 0:
            query_idx += 1
            return "train"
        elif query_idx == 1:
            query_idx += 1
            return "valid"
        elif query_idx == 2:
            query_idx += 1
            return "test"
        else:
            return "quit"
    fake_query.side_effect = query_loader
    thelper.cli.visualize_data(simple_config)
    assert fake_draw.call_count == 104
    assert fake_imread.call_count == 200


@pytest.fixture
def annot_config(request):
    def fin():
        shutil.rmtree(test_create_simple_path, ignore_errors=True)
        shutil.rmtree(test_create_simple_images_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(test_create_simple_images_path, exist_ok=True)
    for cls in range(10):
        os.makedirs(os.path.join(test_create_simple_images_path, str(cls)), exist_ok=True)
        for idx in range(10):
            open(os.path.join(test_create_simple_images_path, str(cls), str(idx) + ".jpg"), "a").close()
    return {
        "name": "simple",
        "bypass_queries": True,
        "datasets": {
            "output": {
                "type": "thelper.data.ImageFolderDataset",
                "params": {
                    "root": test_create_simple_images_path
                }
            }
        },
        "annotator": {
            "type": "thelper.gui.ImageSegmentAnnotator",
            "params": {
                "sample_input_key": "image",
                "labels": [
                    {"id": 255, "name": "foreground", "color": [0, 0, 255]}
                ]
            }
        }
    }


def test_annotate_data(annot_config, mocker):
    import thelper.gui
    fake_run = mocker.patch.object(thelper.gui.ImageSegmentAnnotator, "run")
    fake_imread = mocker.patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    _ = mocker.patch("cv2.setMouseCallback")
    _ = mocker.patch("cv2.namedWindow")
    _ = mocker.patch("cv2.waitKey")
    _ = mocker.patch("cv2.imshow")
    _ = mocker.patch("thelper.gui.create_key_listener")
    fake_config = copy.deepcopy(annot_config)
    del fake_config["name"]
    with pytest.raises(AssertionError):
        thelper.cli.annotate_data(fake_config, test_save_path)
    assert fake_run.call_count == 0
    thelper.cli.annotate_data(annot_config, test_save_path)
    assert fake_run.call_count == 1
    assert fake_imread.call_count == 1


@pytest.fixture
def split_config(request):
    def fin():
        shutil.rmtree(test_create_simple_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(test_create_simple_images_path, exist_ok=True)
    for cls in range(10):
        os.makedirs(os.path.join(test_create_simple_images_path, str(cls)), exist_ok=True)
        for idx in range(10):
            open(os.path.join(test_create_simple_images_path, str(cls), str(idx) + ".jpg"), "a").close()
    return {
        "name": "simple",
        "bypass_queries": True,
        "datasets": {
            "dset": {
                "type": "thelper.data.ImageFolderDataset",
                "params": {
                    "root": test_create_simple_images_path
                }
            }
        },
        "loaders": {
            "shuffle": True,
            "workers": 0,
            "batch_size": 1,
            "skip_class_balancing": True,
            "train_split": {
                "dset": 0.9
            },
            "valid_split": {
                "dset": 0.1
            }
        }
    }


def test_split_data(split_config, mocker):
    fake_config = copy.deepcopy(split_config)
    fake_create = mocker.patch("thelper.data.create_hdf5")
    del fake_config["name"]
    with pytest.raises(AssertionError):
        thelper.cli.split_data(fake_config, test_save_path)
    fake_config = copy.deepcopy(split_config)
    del fake_config["loaders"]
    with pytest.raises(AssertionError):
        thelper.cli.split_data(fake_config, test_save_path)
    fake_config = copy.deepcopy(split_config)
    fake_config["split"] = "dummy"
    with pytest.raises(AssertionError):
        thelper.cli.split_data(fake_config, test_save_path)
    fake_config = copy.deepcopy(split_config)
    fake_config["split"] = {"compression": "dummy"}
    with pytest.raises(AssertionError):
        thelper.cli.split_data(fake_config, test_save_path)
    thelper.cli.split_data(split_config, test_save_path)
    assert fake_create.call_count == 1


@pytest.fixture
def export_config(request):
    def fin():
        shutil.rmtree(test_create_simple_path, ignore_errors=True)
        shutil.rmtree(test_create_simple_images_path, ignore_errors=True)

    fin()
    request.addfinalizer(fin)
    os.makedirs(test_create_simple_images_path, exist_ok=True)
    for cls in range(10):
        os.makedirs(os.path.join(test_create_simple_images_path, str(cls)), exist_ok=True)
        for idx in range(10):
            open(os.path.join(test_create_simple_images_path, str(cls), str(idx) + ".jpg"), "a").close()
    return {
        "name": "simple",
        "bypass_queries": True,
        "model": {
            "type": "torchvision.models.resnet.resnet18",
            "params": {
                "pretrained": True
            }
        },
        "export": {
            "ckpt_name": "test-resnet18-imagenet.pth",
            "trace_name": "test-resnet18-imagenet.zip",
            "save_raw": True,
            "trace_input": "torch.rand(1, 3, 224, 224)",
            "task": {
                "type": "thelper.tasks.Classification",
                "params": {
                    "class_names": "tests/utils/imagenet_classes.json",
                    "input_key": "0",
                    "label_key": "1"
                }
            }
        }
    }


def test_export_model_trace(export_config):
    fake_config = copy.deepcopy(export_config)
    del fake_config["name"]
    with pytest.raises(AssertionError):
        thelper.cli.export_model(fake_config, test_save_path)
    fake_config = copy.deepcopy(export_config)
    fake_config["export"] = "dummy"
    with pytest.raises(AssertionError):
        thelper.cli.export_model(fake_config, test_save_path)
    thelper.cli.export_model(export_config, test_save_path)
    export_ckpt_path = os.path.join(test_create_simple_path, export_config["export"]["ckpt_name"])
    assert os.path.isfile(export_ckpt_path)
    ckptdata = thelper.utils.load_checkpoint(export_ckpt_path)
    model = thelper.nn.create_model(config=None, task=None, ckptdata=ckptdata)
    assert model(eval(export_config["export"]["trace_input"])).shape == (1, 1000)


def test_export_model_weights(export_config):
    export_config = copy.deepcopy(export_config)
    del export_config["export"]["trace_input"]
    del export_config["export"]["trace_name"]
    del export_config["export"]["task"]
    export_config["datasets"] = {
        "dset": {
            "type": "thelper.data.ImageFolderDataset",
            "params": {
                "root": test_create_simple_images_path
            }
        }
    }
    thelper.cli.export_model(export_config, test_save_path)
    export_ckpt_path = os.path.join(test_create_simple_path, export_config["export"]["ckpt_name"])
    assert os.path.isfile(export_ckpt_path)
    ckptdata = thelper.utils.load_checkpoint(export_ckpt_path)
    model = thelper.nn.create_model(config=None, task=None, ckptdata=ckptdata)
    assert model(torch.rand(1, 3, 224, 224)).shape == (1, 10)

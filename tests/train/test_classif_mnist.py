import copy
import os
from typing import Any, AnyStr, Optional  # noqa: F401

import numpy as np
import torch

import thelper

from tests.train.train_utils import (  # noqa: F401 isort:skip
    mnist_config, test_classif_mnist_ft_path, test_classif_mnist_name,
    test_classif_mnist_path, test_save_path
)

config = mnist_config


def test_reload(config):
    train_outputs = thelper.cli.create_session(config, test_save_path)
    assert len(train_outputs) == 2
    assert train_outputs[0]["train/metrics"]["accuracy"] < train_outputs[1]["train/metrics"]["accuracy"]
    ckptdata = thelper.utils.load_checkpoint(test_classif_mnist_path, always_load_latest=True)
    override_config = copy.deepcopy(config)
    override_config["trainer"]["epochs"] = 3
    resume_outputs = thelper.cli.resume_session(ckptdata, save_dir=test_save_path, config=override_config)
    assert len(resume_outputs) == 3
    assert train_outputs[1]["train/metrics"]["accuracy"] < resume_outputs[2]["train/metrics"]["accuracy"]
    ckptdata = thelper.utils.load_checkpoint(test_classif_mnist_path)
    eval_outputs = thelper.cli.resume_session(ckptdata, save_dir=test_save_path, eval_only=True)
    assert any(["test/metrics" in v for v in eval_outputs.values()])
    override_config["trainer"]["epochs"] = 1
    override_config["model"] = {"ckptdata": test_classif_mnist_path}
    override_config["name"] += "-finetune"
    finetune_outputs = thelper.cli.create_session(override_config, test_save_path)
    assert len(finetune_outputs) == 1
    assert finetune_outputs[0]["train/metrics"]["accuracy"] > train_outputs[1]["train/metrics"]["accuracy"]


def compare_dictionaries(dictA, dictB, dictA_name="A", dictB_name="B", path=""):  # pragma: no cover
    err, key_err, value_err = "", "", ""
    old_path = path
    for k in dictA.keys():
        path = old_path + "[%s]" % k
        if k not in dictB:
            key_err += "key %s%s not in %s\n" % (dictB_name, path, dictB_name)
        else:
            if isinstance(dictA[k], dict) and isinstance(dictB[k], dict):
                err += compare_dictionaries(dictA[k], dictB[k], dictA_name, dictB_name, path)
            else:
                if dictA[k] != dictB[k]:
                    value_err += "value of %s%s (%s) not same as %s%s (%s)\n"\
                        % (dictA_name, path, dictA[k], dictB_name, path, dictB[k])
    for k in dictB.keys():
        path = old_path + "[%s]" % k
        if k not in dictA:
            key_err += "key %s%s not in %s\n" % (dictB_name, path, dictA_name)
    return key_err + value_err + err


def test_outputs(config):
    override_config = copy.deepcopy(config)
    override_config["trainer"]["use_tbx"] = True
    train_outputs = thelper.cli.create_session(override_config, test_save_path)
    assert len(train_outputs) == 2
    assert train_outputs[0]["train/metrics"]["accuracy"] < train_outputs[1]["train/metrics"]["accuracy"]
    output_path = os.path.join(test_classif_mnist_path, "output", test_classif_mnist_name)
    assert os.path.isdir(output_path)
    out_dirs = next(os.walk(output_path))[1]
    assert any([out_dir.startswith("train-") for out_dir in out_dirs])
    assert any([out_dir.startswith("valid-") for out_dir in out_dirs])
    for out_dir in out_dirs:
        if out_dir.startswith("test-"):
            continue  # we did not test, skip
        ltype = "train" if out_dir.startswith("train-") else "valid"
        epoch_out_path = os.path.join(output_path, out_dir)
        assert os.path.isdir(epoch_out_path)
        epoch_out_files = next(os.walk(epoch_out_path))[2]
        assert len(epoch_out_files) == 4
        assert "accuracy-0000.txt" in epoch_out_files
        assert "accuracy-0001.txt" in epoch_out_files
        assert "config.json" in epoch_out_files
        assert any([p.startswith("events.out.tfevents.") for p in epoch_out_files])
        for filename in epoch_out_files:
            if filename.startswith("accuracy-"):
                epoch = int(filename.split("-")[1].split(".")[0])
                with open(os.path.join(epoch_out_path, filename), "r") as fd:
                    assert np.isclose(float(fd.readline()), train_outputs[epoch][ltype + "/metrics"]["accuracy"])
            elif filename == "config.json":
                backup_config_path = os.path.join(epoch_out_path, filename)
                backup_config = thelper.utils.load_config(backup_config_path, add_name_if_missing=False)
                assert compare_dictionaries(backup_config, override_config) == ""


# noinspection PyUnusedLocal
def callback(task,          # type: thelper.tasks.utils.Task
             input,         # type: thelper.typedefs.InputType
             pred,          # type: thelper.typedefs.AnyPredictionType
             target,        # type: thelper.typedefs.AnyTargetType
             sample,        # type: thelper.typedefs.SampleType
             loss,          # type: Optional[float]
             iter_idx,      # type: int
             max_iters,     # type: int
             epoch_idx,     # type: int
             max_epochs,    # type: int
             output_path,   # type: AnyStr
             **kwargs,      # type: Any
             # see `thelper.typedefs.IterCallbackParams` for more info
             ):             # type: (...) -> None
    assert isinstance(task, thelper.tasks.Classification)
    assert isinstance(input, torch.Tensor)
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(sample, dict)
    assert isinstance(iter_idx, int)
    assert isinstance(max_iters, int) and iter_idx < max_iters
    assert isinstance(epoch_idx, int)
    assert isinstance(max_epochs, int) and epoch_idx < max_epochs
    assert "hello" in kwargs
    kwargs["hello"][0] = "bye"


def test_callbacks(config, mocker):
    override_config = copy.deepcopy(config)
    callback_kwargs = {"hello": ["hi"]}
    override_config["trainer"]["callback"] = {"type": callback, "params": callback_kwargs}
    override_config["trainer"]["display"] = True
    fake_draw = mocker.patch("thelper.draw.draw")
    assert thelper.cli.create_session(override_config, test_save_path)
    assert fake_draw.call_count > 0
    assert callback_kwargs["hello"][0] == "bye"

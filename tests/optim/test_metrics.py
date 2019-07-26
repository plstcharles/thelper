import numpy as np
import torch

import thelper


def test_accuracy_val():
    batch_size = 32
    num_classes = 10
    target_count = 128
    np.random.seed()
    targets = np.asarray([idx % num_classes for idx in range(target_count)])
    preds = np.random.rand(128, 10)
    nb_correct = np.count_nonzero(targets == np.argmax(preds, axis=1))
    metric = thelper.optim.metrics.Accuracy()
    iter_count = target_count // batch_size
    for iter_idx in range(iter_count):
        metric.update(task=None, input=None,
                      pred=torch.from_numpy(preds[iter_idx * batch_size:(iter_idx + 1) * batch_size]),
                      target=torch.from_numpy(targets[iter_idx * batch_size:(iter_idx + 1) * batch_size]),
                      sample=None, loss=None, iter_idx=iter_idx, max_iters=iter_count, epoch_idx=0, max_epochs=1)
    assert np.isclose(metric.eval(), (nb_correct / target_count) * 100)
    # try looping over, max window should be just the right size
    for iter_idx in range(iter_count):
        metric.update(task=None, input=None,
                      pred=torch.from_numpy(preds[iter_idx * batch_size:(iter_idx + 1) * batch_size]),
                      target=torch.from_numpy(targets[iter_idx * batch_size:(iter_idx + 1) * batch_size]),
                      sample=None, loss=None, iter_idx=iter_idx, max_iters=iter_count, epoch_idx=0, max_epochs=1)
    assert np.isclose(metric.eval(), (nb_correct / target_count) * 100)


def test_accuracy_1d(mocker):
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    class_count = 10
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    metric_config = {"metric": {
        "type": "thelper.optim.metrics.Accuracy",
        "params": {"top_k": 3}
    }}
    metrics = thelper.optim.create_metrics(metric_config)
    metric = metrics["metric"]
    assert isinstance(metric, thelper.optim.metrics.Accuracy)
    assert metric.goal == thelper.optim.Metric.maximize
    assert metric.top_k == 3
    with mocker.patch.object(thelper.optim.metrics.logger, "warning"):
        assert metric.eval() == 0.0
    assert repr(metric)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size,)))
        preds.append(torch.rand((curr_batch_size, class_count)))
        metric.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                      {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                      None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    res = metric.eval()
    assert res is not None and isinstance(res, float) and 0 <= res <= 100
    metric.reset()
    assert metric.eval() == 0.0
    tot_idx = 0
    for iter_idx in range(iter_count):
        metric.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                      {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                      None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert metric.eval() == res


def test_accuracy_nd(mocker):
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    output_shape = (16, 16)
    class_count = 10
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    metric_config = {"metric": {
        "type": "thelper.optim.metrics.Accuracy",
        "params": {"top_k": 3}
    }}
    metrics = thelper.optim.create_metrics(metric_config)
    metric = metrics["metric"]
    assert isinstance(metric, thelper.optim.metrics.Accuracy)
    assert metric.goal == thelper.optim.Metric.maximize
    assert metric.top_k == 3
    with mocker.patch.object(thelper.optim.metrics.logger, "warning"):
        assert metric.eval() == 0.0
    assert repr(metric)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size, *output_shape)))
        preds.append(torch.rand((curr_batch_size, class_count, *output_shape)))
        metric.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                      {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                      None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    res = metric.eval()
    assert res is not None and isinstance(res, float) and 0 <= res <= 100
    metric.reset()
    assert metric.eval() == 0.0
    tot_idx = 0
    for iter_idx in range(iter_count):
        metric.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                      {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                      None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert metric.eval() == res


def test_mae_mse(mocker):
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    metric_config = {"mae": {"type": "thelper.optim.metrics.MeanAbsoluteError"},
                     "mse": {"type": "thelper.optim.metrics.MeanSquaredError"}}
    metrics = thelper.optim.create_metrics(metric_config)
    mae, mse = metrics["mae"], metrics["mse"]
    assert isinstance(mae, thelper.optim.metrics.MeanAbsoluteError)
    assert isinstance(mse, thelper.optim.metrics.MeanSquaredError)
    assert mae.goal == thelper.optim.Metric.minimize
    assert mse.goal == thelper.optim.Metric.minimize
    with mocker.patch.object(thelper.optim.metrics.logger, "warning"):
        assert mae.eval() == 0.0
        assert mse.eval() == 0.0
    assert repr(mae) and repr(mse)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.randn((curr_batch_size,)))
        preds.append(torch.randn((curr_batch_size, )))
        mae.update(None, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                   {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                   None, iter_idx, iter_count, 0, 1)
        mse.update(None, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                   {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                   None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    mae_res, mse_res = mae.eval(), mse.eval()
    assert mae_res is not None and isinstance(mae_res, float) and mae_res >= 0
    assert mse_res is not None and isinstance(mse_res, float) and mse_res >= 0
    assert mse_res >= mae_res
    mae.reset()
    mse.reset()
    assert mae.eval() == 0.0
    assert mse.eval() == 0.0
    tot_idx = 0
    for iter_idx in range(iter_count):
        mae.update(None, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                   {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                   None, iter_idx, iter_count, 0, 1)
        mse.update(None, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                   {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                   None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert mae.eval() == mae_res
    assert mse.eval() == mse_res


def test_external_metrics():
    batch_size = 32
    iter_count = 64
    class_count = 7
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    metric_config = {
        "f1": {
            "type": "thelper.optim.metrics.ExternalMetric",
            "params": {
                "metric_name": "sklearn.metrics.f1_score",
                "metric_params": {},
                "metric_type": "classif_top1",
                "target_name": "0",
                "metric_goal": "max"
            }
        },
        "auc": {
            "type": "thelper.optim.metrics.ExternalMetric",
            "params": {
                "metric_name": "sklearn.metrics.roc_auc_score",
                "metric_params": {},
                "metric_type": "classif_scores",
                "target_name": "1",
                "metric_goal": "max",
                "class_names": class_names
            }
        },
    }
    metrics = thelper.optim.create_metrics(metric_config)
    metric_f1, metric_auc = metrics["f1"], metrics["auc"]
    assert isinstance(metric_f1, thelper.optim.metrics.ExternalMetric)
    assert isinstance(metric_auc, thelper.optim.metrics.ExternalMetric)
    assert metric_f1.goal == thelper.optim.Metric.maximize
    assert metric_auc.goal == thelper.optim.Metric.maximize
    assert repr(metric_f1) and repr(metric_auc)
    targets, preds = [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size,)))
        preds.append(torch.rand((curr_batch_size, class_count)))
        metric_f1.update(task, None, preds[iter_idx], targets[iter_idx],
                         {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                         None, iter_idx, iter_count, 0, 1)
        metric_auc.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    f1_res, auc_res = metric_f1.eval(), metric_auc.eval()
    assert f1_res is not None and isinstance(f1_res, float) and 0 <= f1_res <= 1
    assert auc_res is not None and isinstance(auc_res, float) and 0 <= auc_res <= 1
    metric_f1.reset()
    tot_idx = 0
    for iter_idx in range(iter_count):
        metric_f1.update(task, None, preds[iter_idx], targets[iter_idx],
                         {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                         None, iter_idx, iter_count, 0, 1)
        metric_auc.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                          None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert metric_f1.eval() == f1_res
    assert metric_auc.eval() == auc_res


def test_roccurve():
    batch_size = 32
    iter_count = 128
    class_count = 3
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    metric_config = {
        "tpr": {
            "type": "thelper.optim.metrics.ROCCurve",
            "params": {
                "target_name": "1",
                "target_fpr": 0.99
            }
        },
        "fpr": {
            "type": "thelper.optim.metrics.ROCCurve",
            "params": {
                "target_name": "0",
                "target_tpr": 0.9
            }
        },
        "auc": {
            "type": "thelper.optim.metrics.ROCCurve",
            "params": {
                "target_name": "2"
            }
        },
    }
    metrics = thelper.optim.create_metrics(metric_config)
    metric_tpr, metric_fpr, metric_auc = metrics["tpr"], metrics["fpr"], metrics["auc"]
    assert isinstance(metric_tpr, thelper.optim.metrics.ROCCurve)
    assert isinstance(metric_fpr, thelper.optim.metrics.ROCCurve)
    assert isinstance(metric_auc, thelper.optim.metrics.ROCCurve)
    assert metric_tpr.goal == thelper.optim.Metric.maximize
    assert metric_fpr.goal == thelper.optim.Metric.minimize
    assert metric_auc.goal == thelper.optim.Metric.maximize
    assert repr(metric_tpr) and repr(metric_fpr) and repr(metric_auc)
    metric_ref = thelper.optim.metrics.ExternalMetric(metric_name="sklearn.metrics.roc_auc_score",
                                                      metric_type="classif_scores",
                                                      target_name="2", metric_goal="max")
    targets, preds = [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size,)))
        preds.append(torch.rand((curr_batch_size, class_count)))
        metric_tpr.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        metric_fpr.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        metric_auc.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        metric_ref.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    tpr_res, fpr_res, auc_res, ref_res = metric_tpr.eval(), metric_fpr.eval(), metric_auc.eval(), metric_ref.eval()

    assert tpr_res is not None and isinstance(tpr_res, float) and 0 <= tpr_res <= 1
    assert fpr_res is not None and isinstance(fpr_res, float) and 0 <= fpr_res <= 1
    assert auc_res is not None and isinstance(auc_res, float) and 0 <= auc_res <= 1
    assert ref_res is not None and isinstance(ref_res, float) and 0 <= ref_res <= 1
    assert np.isclose(ref_res, auc_res)
    metric_tpr.reset()
    metric_fpr.reset()
    metric_auc.reset()
    tot_idx = 0
    for iter_idx in range(iter_count):
        metric_tpr.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        metric_fpr.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        metric_auc.update(task, None, preds[iter_idx], targets[iter_idx],
                          {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                          None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert metric_tpr.eval() == tpr_res
    assert metric_fpr.eval() == fpr_res
    assert metric_auc.eval() == auc_res


def test_psnr(mocker):
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    metric_config = {"psnr": {"type": "thelper.optim.metrics.PSNR"}}
    metric_psnr = thelper.optim.create_metrics(metric_config)["psnr"]
    assert isinstance(metric_psnr, thelper.optim.metrics.PSNR)
    assert metric_psnr.goal == thelper.optim.Metric.maximize
    with mocker.patch.object(thelper.optim.metrics.logger, "warning"):
        assert metric_psnr.eval() == 0.0
    assert repr(metric_psnr)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.rand((curr_batch_size,)))
        preds.append(torch.rand((curr_batch_size,)))
        metric_psnr.update(None, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                           {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                           None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    psnr_res = metric_psnr.eval()
    assert psnr_res is not None and isinstance(psnr_res, float) and psnr_res >= 0
    metric_psnr.reset()
    assert metric_psnr.eval() == 0.0
    tot_idx = 0
    for iter_idx in range(iter_count):
        metric_psnr.update(None, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                           {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                           None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert metric_psnr.eval() == psnr_res

import numpy as np
import torch

import thelper


def test_classif_logger():
    # classification results are expected in 1D format; lets build some dummy data...
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    class_count = 10
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    consumer_config = {"consumer": {
        "type": "thelper.train.utils.ClassifLogger",
        "params": {"top_k": 3, "report_count": 10,
                   "class_names": class_names, "log_keys": ["idx"]}
    }}
    consumers = thelper.train.create_consumers(consumer_config)
    consumer = consumers["consumer"]
    assert isinstance(consumer, thelper.train.utils.ClassifLogger)
    assert consumer.top_k == 3
    assert consumer.report_count == 10
    assert consumer.class_names == class_names
    assert consumer.report() is None
    assert repr(consumer)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size, )))
        preds.append(torch.rand((curr_batch_size, class_count)))
        consumer.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                        {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                        None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    report = consumer.report()
    assert report is not None and isinstance(report, str)
    assert len(report.split("\n")) == 11  # 10 lines + header
    assert "target_name,target_score,pred_1_name,pred_1_score,pred_2_name," \
           "pred_2_score,pred_3_name,pred_3_score,idx" == report.split("\n")[0]
    consumer.reset()
    assert consumer.report() is None
    consumer.class_names = None
    tot_idx = 0
    for iter_idx in range(iter_count):
        consumer.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                        {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                        None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert consumer.report() == report


def test_classif_report():
    # classification results are expected in 1D format; lets build some dummy data...
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    class_count = 10
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    consumer_config = {"consumer": {
        "type": "thelper.train.utils.ClassifReport",
        "params": {"class_names": class_names}
    }}
    consumers = thelper.train.create_consumers(consumer_config)
    consumer = consumers["consumer"]
    assert isinstance(consumer, thelper.train.utils.ClassifReport)
    assert consumer.class_names == class_names
    assert consumer.report() is None
    assert repr(consumer)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size, )))
        preds.append(torch.rand((curr_batch_size, class_count)))
        consumer.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                        {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                        None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    report = consumer.report()
    assert report is not None and isinstance(report, str)
    assert report.endswith(f"{tot_idx}\n")  # should be total number of samples in last cell
    consumer.reset()
    assert consumer.report() is None
    consumer.class_names = None
    tot_idx = 0
    for iter_idx in range(iter_count):
        consumer.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                        {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                        None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert consumer.report() == report


def test_confmat():
    # classification results are expected in 1D format; lets build some dummy data...
    batch_size = 16
    iter_count = 32
    input_shape = (3, 32, 32)
    class_count = 10
    class_names = [str(i) for i in range(class_count)]
    task = thelper.tasks.Classification(class_names, "input", "gt", ["idx"])
    consumer_config = {"consumer": {
        "type": "thelper.train.utils.ConfusionMatrix",
        "params": {"class_names": class_names}
    }}
    consumers = thelper.train.create_consumers(consumer_config)
    consumer = consumers["consumer"]
    assert isinstance(consumer, thelper.train.utils.ConfusionMatrix)
    assert consumer.class_names == class_names
    assert consumer.report() is None
    assert repr(consumer)
    inputs, targets, preds = [], [], []
    tot_idx = 0
    for iter_idx in range(iter_count):
        # set batch size to one for 'lingering' sample in last minibatch
        curr_batch_size = batch_size if iter_idx < iter_count - 1 else 1
        inputs.append(torch.randn((curr_batch_size, *input_shape)))
        targets.append(torch.randint(low=0, high=class_count, size=(curr_batch_size, )))
        preds.append(torch.rand((curr_batch_size, class_count)))
        consumer.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                        {"idx": [tot_idx + idx for idx in range(curr_batch_size)]},
                        None, iter_idx, iter_count, 0, 1)
        tot_idx += curr_batch_size
    report = consumer.report()
    assert report is not None and isinstance(report, str)
    assert report.endswith(f"{tot_idx}\n")  # should be total number of samples in last cell
    render = consumer.render()
    assert render is None or isinstance(render, np.ndarray)
    consumer.reset()
    assert consumer.report() is None
    consumer.class_names = None
    tot_idx = 0
    for iter_idx in range(iter_count):
        consumer.update(task, inputs[iter_idx], preds[iter_idx], targets[iter_idx],
                        {"idx": [tot_idx + idx for idx in range(targets[iter_idx].shape[0])]},
                        None, iter_idx, iter_count, 0, 1)
        tot_idx += targets[iter_idx].shape[0]
    assert consumer.report() == report

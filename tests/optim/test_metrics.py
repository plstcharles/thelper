import unittest

from torch.tensor import Tensor

from thelper.optim.metrics import RawPredictions


class TestMetrics(unittest.TestCase):
    def test_raw_predictions_metric_basic_use_case(self):
        """Validations for basic use case:
            - predictions are gradually saved on 'accumulate' calls
            - non-serializable tensors are converted to JSON lists
            - meta information are transferred accordingly to matching predictions
        """

        metric = RawPredictions()
        list1d = [1., 2., 3., 4., 5., 6.]
        list2d = [[1., 2.], [3., 4.], [5., 6.]]
        list3d = [[[1.], [2.]], [[3.], [4.]], [[5.], [6.]]]

        gt = ['class1', 'class2']
        pred = Tensor(list2d)   # predictions of sample batch
        meta = {
            'l1d': list1d,
            'l2d': list2d,
            'l3d': list3d,
        }

        # should accumulate and convert to serializable types
        metric.accumulate(pred, gt, meta)
        assert isinstance(metric.predictions, list)
        assert len(metric.predictions) == len(pred), "saved predictions should match dimensions"
        for mp, tp in zip(metric.predictions, pred):
            assert isinstance(mp, dict), "predictions should be a dictionary of information"
            assert isinstance(mp['predictions'], list), "predictions tensor should be converted to serializable list"
            for mp_pred, tp_pred in zip(mp['predictions'], tp):
                assert mp_pred == tp_pred, "corresponding predictions should be transferred to serializable list"
            for k in meta.keys():
                assert k in mp, "all meta info should be added"
                assert isinstance(mp[k], list), "no tensors should be added (non-serializable)"

        # second call appends more predictions
        metric.accumulate(pred, gt, meta)
        assert len(metric.predictions) == len(pred) * 2, "saved predictions should be extended with new ones"
        for mp, tp in zip(metric.predictions, list(pred) + list(pred)):
            for mp_pred, tp_pred in zip(mp['predictions'], tp):
                assert mp_pred == tp_pred, "corresponding predictions should be transferred to serializable list"

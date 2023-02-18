from typing import Union
from metrics.metric_abstract import MetricAbstract, MetricCI_clopper_pearson
from metrics.metric_classification import TPR, Recall
from data_slicing.data_slice import DataSlice
import json


class MultithresholdMetricCI_clopper_pearson:

    @classmethod
    def calc_CI(cls, s: DataSlice, Y: tuple, metric_CI: Union[MetricCI_clopper_pearson, MetricAbstract], *args, **kwargs) -> DataSlice:
        # unravel metric values
        metric_values = json.loads(s.get_metric_value(cls.name))
        thresholds = metric_values['thresholds']

        metric_values[f'{metric_CI.name}_LB'] = []
        metric_values[f'{metric_CI.name}_UB'] = []
        s_ = s.deepcopy(clean=True)

        def _calc_CI(t):
            Y_t = (Y[0], Y[1] > t)
            s_t = s_.deepcopy()
            s_t = metric_CI.metric_function(s_t, Y_t)
            s_t = metric_CI.calc_CI(s_t)
            return s_t.get_metric_value(f'{metric_CI.name}_LB'), s_t.get_metric_value(f'{metric_CI.name}_UB')

        ci = map(_calc_CI, thresholds)
        for ci_LB, ci_UB in ci:
            metric_values[f'{metric_CI.name}_LB'].append(ci_LB)
            metric_values[f'{metric_CI.name}_UB'].append(ci_UB)

        s.update_metric_value(cls.name, json.dumps(metric_values))
        return s


class ROC(MetricAbstract, MultithresholdMetricCI_clopper_pearson):
    """ROC Curve"""
    from sklearn.metrics import roc_curve
    name = 'ROC'

    @classmethod
    def _ROC(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        fpr, tpr, thresholds = cls.roc_curve(*Y)
        metric_value = json.dumps({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()})
        s.update_metric_value(cls.name, metric_value)
        return s
    m = _ROC

    @classmethod
    def calc_CI(cls, s: DataSlice, Y, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, Y, TPR)


class PRC(MetricAbstract, MultithresholdMetricCI_clopper_pearson):
    """Precision Recall Curve"""
    from sklearn.metrics import precision_recall_curve
    name = 'PRC'

    @classmethod
    def _PRC(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        precision, recall, thresholds = cls.precision_recall_curve(*Y)
        metric_value = json.dumps({"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()})
        s.update_metric_value(cls.name, metric_value)
        return s
    m = _PRC

    @classmethod
    def calc_CI(cls, s: DataSlice, Y, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, Y, Recall)

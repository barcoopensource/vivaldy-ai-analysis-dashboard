from typing import Dict, List, Tuple
from data_slicing.data_slice import DataSlice
from metrics.evaluation.metrics_evaluation_abstract import MetricsEvaluationAbstract
from metrics.metric_abstract import MetricAbstract, MetricSingleSlice


class MetricsEvaluationBinaryClassification(MetricsEvaluationAbstract):
    def process(self, slices: List[DataSlice], Y: List, *args, **kwargs):
        return super().process(slices, Y, *args, **kwargs)

    @classmethod
    def setup(cls, slices: List[DataSlice], Y: List, *args, **kwargs):
        super().setup(slices, Y, *args, **kwargs)
        cls.classes = cls.settings.get('evaluation').get('binary_classification').get('classes')

        # intialize metric classes
        cls._initialize_metrics(cls.settings.get('evaluation').get('binary_classification').get('metrics').keys())
        # metric: MetricAbstract
        # for metric in cls.settings.get('evaluation').get('binary_classification').get('metrics').keys():
        #     metric.setup(cls.settings)
        return slices

    @classmethod
    def evaluate_metrics(cls, slices: List[DataSlice], Y: List, **kwargs):
        slices = list(cls.starmap(cls._evaluate_metrics, zip(slices, Y)))
        return slices

    @classmethod
    def _evaluate_metrics(cls, s: DataSlice, Y: Tuple):
        minimum_samples = cls.settings.get('evaluation').get('slice_evaluation_minimum_samples')
        metrics_to_evaluate = cls.settings.get('evaluation').get('binary_classification').get('metrics')
        if s.size < minimum_samples:
            return s
        for metric, metric_settings in metrics_to_evaluate.items():
            metric: MetricSingleSlice
            s = metric.metric_function(s, Y)
            if s is None:
                print('stop')
            if metric_settings.get('CI'):
                s = metric.calc_CI(s, Y)
        return s


class MetricsEvaluationBinaryClassificationMultithreshold(MetricsEvaluationBinaryClassification):
    @classmethod
    def setup(cls, slices: List[DataSlice], Y: List, *args, **kwargs):
        super().setup(slices, Y, *args, **kwargs)
        cls._initialize_metrics(cls.settings.get('evaluation').get('binary_classification').get('metrics_multi_threshold').keys())
        # metric: MetricAbstract
        # for metric in cls.settings.get('evaluation').get('binary_classification').get('metrics_multi_threshold').keys():
        #     metric.setup(cls.settings)
        return slices

    @classmethod
    def _evaluate_metrics(cls, s: DataSlice, Y: Tuple):
        minimum_samples = cls.settings.get('evaluation').get('slice_evaluation_minimum_samples')
        metrics_to_evaluate = cls.settings.get('evaluation').get('binary_classification').get('metrics_multi_threshold')
        if s.size < minimum_samples:
            return s
        for metric, metric_settings in metrics_to_evaluate.items():
            s = metric.m(s, Y)
            if metric_settings.get('CI'):
                s = metric.calc_CI(s, Y)
        return s

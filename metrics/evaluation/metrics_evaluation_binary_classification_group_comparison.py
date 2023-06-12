from typing import Dict, List, Tuple
from data_slicing.data_slice import DataSlice
import pandas as pd
from metrics.evaluation.metrics_evaluation_group_comparison_abstract import MetricsEvaluationGroupComparisonAbstract
from metrics.metric_group_comparison import MetricGroupComparisonAbstract


class MetricsEvaluationBinaryClassificationGroupComparison(MetricsEvaluationGroupComparisonAbstract):
    def process(self, slices: List[DataSlice], ref_slices: List[DataSlice], Y: List, ref_Y: List, *args, **kwargs):
        return super().process(slices, ref_slices, Y, ref_Y, *args, **kwargs)

    @classmethod
    def setup(cls, slices: List[DataSlice], ref_slices: List[DataSlice], Y: List, ref_Y: List, *args, **kwargs):
        super().setup(slices, ref_slices, Y, ref_Y, *args, **kwargs)
        return slices, ref_slices

    @classmethod
    def evaluate_metrics(cls, slices: List[DataSlice], ref_slices: List[DataSlice], Y: List, ref_Y: List, **kwargs):
        slices, ref_slices = zip(*cls.starmap(cls._evaluate_metrics, zip(slices, ref_slices, Y, ref_Y)))
        return slices, ref_slices

    @classmethod
    def _evaluate_metrics(cls, s: DataSlice, ref_s: DataSlice, Y: Tuple, ref_Y: Tuple):
        minimum_samples = cls.settings.get('evaluation').get('slice_evaluation_minimum_samples')
        if s.size < minimum_samples or ref_s.size < minimum_samples:
            return s, ref_s
        metrics_to_evaluate = cls.settings.get('evaluation').get('binary_classification').get('metrics_group_comparison')
        for metric, metric_settings in metrics_to_evaluate.items():
            metric: MetricGroupComparisonAbstract
            s, ref_s = metric.metric_function(s, ref_s, Y, ref_Y)
            if metric_settings.get('CI'):
                s, ref_s = metric.calc_CI(s, ref_s, Y, ref_Y)
        return s, ref_s

    @staticmethod
    def get_complementary_slices(slices: List[DataSlice], y_predict: pd.Series) -> List[DataSlice]:
        ref_slices = list(DataSlice.from_dataframe(y_predict[~y_predict.index.isin(s)]) for s in slices)
        return ref_slices

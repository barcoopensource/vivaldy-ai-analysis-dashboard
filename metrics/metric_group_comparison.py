from metrics.metric_abstract import MetricAbstract
from metrics.metric_abstract import MetricCI_clopper_pearson, MetricCI_bootstrap, MetricCI_given, MetricCI_normal, MetricCI_none
from metrics.metric_abstract import MetricChained
from data_slicing.data_slice import DataSlice
from metrics.metric_classification import *
import pandas as pd
from typing import Tuple


class MetricGroupComparisonAbstract(MetricAbstract):
    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, Y: tuple, ref_Y: tuple, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        print("Warning: This function needs to be inherited")
        pass


class MetricGroupComparisonUnlabelledAbstract(MetricAbstract):
    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, y: pd.Series, ref_y: pd.Series, column_name: str, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        print("Warning: This function needs to be inherited")
        pass


class MetricCI_GroupComparison_bootstrap():
    @staticmethod
    def clean_slice(s: DataSlice) -> DataSlice:
        return s.deepcopy(clean=True)

    @classmethod
    def calc_CI(cls, s: DataSlice, ref_s: DataSlice, Y: tuple, ref_Y: tuple, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        n_resamples = cls.settings.get('evaluation').get('binary_classification').get('n_resamples')
        from scipy.stats import bootstrap
        from functools import partial
        import numpy as np
        from scipy.stats._common import ConfidenceInterval

        # Dummy estimation of CI based on alternatively fixing population and reference, giving two sigma's
        # The final sigma can then be estimated by weighing these two values according to sample size

        _bootstrap_function_1 = partial(cls._bootstrap_function, s=cls.clean_slice(s), ref_s=cls.clean_slice(ref_s), ref_Y=ref_Y)
        res_bootstrap_1 = bootstrap(Y, _bootstrap_function_1, n_resamples=n_resamples, vectorized=False, method='percentile', paired=True)
        ci_1: ConfidenceInterval = res_bootstrap_1.confidence_interval

        _bootstrap_function_2 = partial(cls._bootstrap_function, s=cls.clean_slice(ref_s), ref_s=cls.clean_slice(s), ref_Y=Y)
        res_bootstrap_2 = bootstrap(ref_Y, _bootstrap_function_2, n_resamples=n_resamples, vectorized=False, method='percentile', paired=True)
        ci_2: ConfidenceInterval = res_bootstrap_2.confidence_interval

        var_1_low = ((s.get_metric_value(cls.name) - ci_1.low) / 2.0) ** 2
        var_1_high = ((ci_1.high - s.get_metric_value(cls.name)) / 2.0) ** 2

        var_2_low = ((-ref_s.get_metric_value(cls.name) - (-ci_2.high)) / 2.0) ** 2
        var_2_high = ((-ci_2.low - (-ref_s.get_metric_value(cls.name))) / 2.0) ** 2

        var_low = var_1_low + var_2_low
        var_high = var_1_high + var_2_high

        ci = ConfidenceInterval(ci_1.low - 2.0 * np.sqrt(var_low), ci_1.high + 2.0 * np.sqrt(var_high))

        s.update_metric_value(cls.name + '_LB', ci.low)
        s.update_metric_value(cls.name + '_UB', ci.high)
        return s, ref_s

    @classmethod
    def _bootstrap_function(cls, *Y, s: DataSlice, ref_s: DataSlice, ref_Y: tuple):
        s, ref_s = cls.metric_function(s.deepcopy(), ref_s.deepcopy(), Y, ref_Y)
        return s.get_metric_value(cls.name)


class StatisticalParityDifference(MetricGroupComparisonAbstract, MetricCI_GroupComparison_bootstrap, MetricChained):
    """Statistical Parity Difference"""
    name = 'StatisticalParityDifference'

    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, Y: tuple, ref_Y: tuple, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        s = cls.chain(s, PredictedPrevalence, Y, *args, **kwargs)
        ref_s = cls.chain(ref_s, PredictedPrevalence, ref_Y, *args, **kwargs)
        d = s.get_metric_value(PredictedPrevalence.name) - ref_s.get_metric_value(PredictedPrevalence.name)
        s.update_metric_value(cls.name, d)
        ref_s.update_metric_value(cls.name, -d)
        return s, ref_s


class DisparateImpact(MetricGroupComparisonAbstract, MetricCI_GroupComparison_bootstrap, MetricChained):
    """Disparate Impact"""
    name = 'DisparateImpact'

    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, Y: tuple, ref_Y: tuple, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        s = cls.chain(s, PredictedPrevalence, Y, *args, **kwargs)
        ref_s = cls.chain(ref_s, PredictedPrevalence, ref_Y, *args, **kwargs)
        d = s.get_metric_value(PredictedPrevalence.name) / ref_s.get_metric_value(PredictedPrevalence.name)
        s.update_metric_value(cls.name, d)
        ref_s.update_metric_value(cls.name, -d)
        return s, ref_s


class TPRDisparity(MetricGroupComparisonAbstract, MetricCI_GroupComparison_bootstrap, MetricChained):
    """True Positive Rate Disparity"""
    name = 'TPR_Disparity'

    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, Y: tuple, ref_Y: tuple, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        s = cls.chain(s, TPR, Y, *args, **kwargs)
        ref_s = cls.chain(ref_s, TPR, ref_Y, *args, **kwargs)
        d = s.get_metric_value(TPR.name) - ref_s.get_metric_value(TPR.name)
        s.update_metric_value(cls.name, d)
        ref_s.update_metric_value(cls.name, -d)
        return s, ref_s


class EqualOpportunityDifference(TPRDisparity):
    """Equal Opportunity Difference"""
    name = 'EqualOpportunityDifference'


class AverageOddsDifference(MetricGroupComparisonAbstract, MetricCI_GroupComparison_bootstrap, MetricChained):
    """Average Odds Difference"""
    name = 'AverageOddsDifference'

    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, Y: tuple, ref_Y: tuple, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        s = cls.chain(s, TPR, Y, *args, **kwargs)
        s = cls.chain(s, FPR, Y, *args, **kwargs)
        ref_s = cls.chain(ref_s, TPR, ref_Y, *args, **kwargs)
        ref_s = cls.chain(ref_s, FPR, ref_Y, *args, **kwargs)
        d = 0.5 * ((s.get_metric_value(FPR.name) - ref_s.get_metric_value(FPR.name)) + (s.get_metric_value(TPR.name) - ref_s.get_metric_value(TPR.name)))
        s.update_metric_value(cls.name, d)
        ref_s.update_metric_value(cls.name, -d)
        return s, ref_s


class KolmogorovSmirnoff(MetricGroupComparisonUnlabelledAbstract, MetricCI_GroupComparison_bootstrap):
    """Kolmogorov Smirnoff test statistic"""
    name = 'kolmogorov_smirnoff'

    @classmethod
    def metric_function(cls, s: DataSlice, ref_s: DataSlice, y: pd.Series, ref_y: pd.Series, column_name: str, *args, **kwargs) -> Tuple[DataSlice, DataSlice]:
        from scipy.stats import ks_2samp

        stat_result = ks_2samp(y, ref_y)._asdict()

        ks_value = stat_result['statistic'] * stat_result['statistic_sign']
        s.update_metric_value(f'{cls.name}_{column_name}_statistic', ks_value)
        ref_s.update_metric_value(f'{cls.name}_{column_name}_statistic', -ks_value)
        return s, ref_s

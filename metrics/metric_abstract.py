from types import FunctionType
from dataclasses import dataclass
from data_slicing.data_slice import DataSlice


@dataclass
class MetricAbstract():
    name: str
    m: FunctionType
    settings: dict

    @classmethod
    def setup(cls, settings):
        cls.settings = settings

    @classmethod
    def metric_function(cls, *args, **kwargs):
        "This function needs to be inherited"
        pass


class MetricCI_clopper_pearson:
    from scipy.stats import binomtest
    from scipy.stats._common import ConfidenceInterval

    @classmethod
    def calc_CI(cls, s: DataSlice, k: str, n: str, *args, **kwargs) -> DataSlice:
        import numpy as np
        k = s.get_metric_value(k)
        n = s.get_metric_value(n)
        if n == 0:
            ci = cls.ConfidenceInterval(low=np.nan, high=np.nan)
        else:
            ci = cls.binomtest(k=k, n=n, p=k / n).proportion_ci(confidence_level=0.95)
        s.update_metric_value(cls.name + '_LB', ci.low)
        s.update_metric_value(cls.name + '_UB', ci.high)
        return s


class MetricCI_bootstrap():
    @staticmethod
    def clean_slice(s: DataSlice) -> DataSlice:
        return s.deepcopy(clean=True)

    @classmethod
    def calc_CI(cls, s: DataSlice, Y, *args, **kwargs) -> DataSlice:
        n_resamples = cls.settings.get('evaluation').get('binary_classification').get('n_resamples')
        from scipy.stats import bootstrap
        from functools import partial

        _bootstrap_function = partial(cls._bootstrap_function, s=cls.clean_slice(s))
        res_bootstrap = bootstrap(Y, _bootstrap_function, n_resamples=n_resamples, vectorized=False, method='percentile', paired=True)
        ci = res_bootstrap.confidence_interval

        s.update_metric_value(cls.name + '_LB', ci.low)
        s.update_metric_value(cls.name + '_UB', ci.high)
        return s

    @classmethod
    def _bootstrap_function(cls, *Y, s: DataSlice,):
        s = cls.metric_function(s.deepcopy(), Y)
        return s.get_metric_value(cls.name)


class MetricCI_given:
    @classmethod
    def calc_CI(cls, s: DataSlice, LB, UB, *args, **kwargs) -> DataSlice:
        s.update_metric_value(cls.name + '_LB', LB)
        s.update_metric_value(cls.name + '_UB', UB)
        return s


class MetricCI_normal(MetricCI_given):
    @classmethod
    def calc_CI(cls, s: DataSlice, loc, std, alpha=0.95, *args, **kwargs) -> DataSlice:
        from scipy import stats
        import numpy as np
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(
            lower_upper_q,
            loc=loc,
            scale=std)

        ci[ci > 1] = 1
        ci[ci < 0] = 0
        return super().calc_CI(s, ci[0], ci[1])


class MetricCI_none:
    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        import numpy as np
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = cls.name
        s.update_metric_value(name + '_LB', np.nan)
        s.update_metric_value(name + '_UB', np.nan)
        return s


class MetricChained:
    @classmethod
    def chain(cls, s: DataSlice, metric: MetricAbstract, *args, **kwargs) -> DataSlice:
        if metric.name not in s.get_all_metrics():
            metric.setup(cls.settings)
            s = metric.metric_function(s, *args, **kwargs)
        return s


class MetricSingleSlice(MetricAbstract):
    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        print("Warning: This function needs to be inherited")
        pass

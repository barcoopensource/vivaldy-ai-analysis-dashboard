from typing import Dict, List, Tuple
from metrics.metric_abstract import MetricAbstract


class MetricsEvaluationAbstract:
    pool = None
    settings: dict

    def __init__(self):
        pass

    @classmethod
    def setup(cls, slices, *args, settings={}, mp_settings={}, **kwargs):
        from itertools import starmap
        import functools
        cls.settings = settings
        if mp_settings.get('use_mp'):
            import multiprocessing
            cls.pool = multiprocessing.Pool(mp_settings.get('N_threads'))
            cls.starmap = functools.partial(cls.pool.starmap, chunksize=mp_settings.get('chunksize'))
            cls.map = functools.partial(cls.pool.imap, chunksize=mp_settings.get('chunksize'))
        else:
            cls.pool = None
            cls.starmap = starmap
            cls.map = map
        pass

    def evaluate_metrics(self, slices, *args, **kwargs):
        pass

    def finalize(self, slices, *args, **kwargs):
        if self.pool is not None:
            self.pool.close()
        return slices

    def process(self, slices, *args, **kwargs):
        print(f'Evaluating metrics {self.__class__.__name__}')
        slices = self.setup(slices, *args, **kwargs)
        slices = self.evaluate_metrics(slices, *args, **kwargs)
        return self.finalize(slices, *args, **kwargs)

    @classmethod
    def _initialize_metrics(cls, metrics: List[MetricAbstract]):
        for metric in metrics:
            metric.setup(cls.settings)

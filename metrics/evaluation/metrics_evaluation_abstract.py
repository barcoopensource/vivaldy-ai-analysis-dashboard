from metrics.metric_abstract import MetricAbstract


class MetricsEvaluationAbstract:
    settings: dict

    def __init__(self):
        pass

    @classmethod
    def _init_from_pool(cls, settings, *args, **kwargs):
        import warnings
        warnings.simplefilter(action='ignore')
        cls.settings = settings
        MetricAbstract.setup(cls.settings)

    @classmethod
    def setup(cls, slices, *args, settings={}, mp_settings={}, **kwargs):
        from itertools import starmap
        import functools
        cls.settings = settings
        if mp_settings.get('use_mp'):
            from multiprocessing import Pool
            cls.pool = Pool(mp_settings.get('N_threads'), initializer=cls._init_from_pool, initargs=(settings,))
            cls.starmap = functools.partial(cls.pool.starmap, chunksize=mp_settings.get('chunksize'))
            cls.map = functools.partial(cls.pool.imap, chunksize=mp_settings.get('chunksize'))
        else:
            cls.pool = None
            cls.starmap = starmap
            cls.map = map

        MetricAbstract.setup(cls.settings)
        return slices

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

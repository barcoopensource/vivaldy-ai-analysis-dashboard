from metrics.evaluation.metrics_evaluation_abstract import MetricsEvaluationAbstract


class MetricsEvaluationGroupComparisonAbstract(MetricsEvaluationAbstract):
    def evaluate_metrics(self, slices, ref_slicess, *args, **kwargs):
        pass

    def finalize(self, slices, ref_slices, *args, **kwargs):
        if self.pool is not None:
            self.pool.close()
        return slices, ref_slices

    def process(self, slices, ref_slices, *args, **kwargs):
        print(f'Evaluating metrics with reference {self.__class__.__name__}')
        slices, ref_slices = self.setup(slices, ref_slices, *args, **kwargs)
        slices, ref_slices = self.evaluate_metrics(slices, ref_slices, *args, **kwargs)
        return self.finalize(slices, ref_slices, *args, **kwargs)

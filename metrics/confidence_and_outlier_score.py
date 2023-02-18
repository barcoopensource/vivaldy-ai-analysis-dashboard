from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_slicing_pipeline.data_slicing_pipeline_abstract import DataSlicingPipelineAbstract
from metrics.metric_abstract import MetricAbstract


class ConfidenceScore(MetricAbstract):
    name = 'confidence_score'

    @classmethod
    def metric_function(cls, sfp: DataSlicingPipelineAbstract, **kwargs):
        import numpy as np

        df_export = sfp.df_export
        df_original = sfp.df_original
        probabilites_columns = kwargs['probabilites_columns']
        df_export[cls.name] = df_original[probabilites_columns].var(axis=1)
        df_export[cls.name] =  -np.log10(df_export[cls.name])

        sfp.df_export = df_export


class OutlierScore(MetricAbstract):
    name = 'outlier_score'

    @classmethod
    def metric_function(cls, sfp: DataSlicingPipelineAbstract, **kwargs):
        df_export = sfp.df_export
        df_original = sfp.df_original
        probabilites_columns = kwargs['probabilites_columns']
        df_export[cls.name] = df_original[probabilites_columns]

        sfp.df_export = df_export

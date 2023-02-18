"""
Add metrics which use a reference slice to calculate the actual metrics, such as difference or distance metrics between two data slices.
Concrete examples include dedicated fairness and bias metrics.
"""
from __future__ import annotations
from postprocessing.postprocessing_abstract import *
from metrics.metric_group_comparison import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_slicing_pipeline.data_slicing_pipeline_abstract import DataSlicingPipelineAbstract


class AddGroupComparisonStatistics(PostProcessingAbstract):

    settings = None

    @classmethod
    def process(cls, sfp: DataSlicingPipelineAbstract, postprocessing_settings: dict) -> pd.DataFrame:
        cls.settings = sfp.settings

        for d, slices in sfp.slices.items():
            if d == 0:
                continue

            statistic: MetricGroupComparisonUnlabelledAbstract
            for statistic in postprocessing_settings['metrics']:
                for c in postprocessing_settings['columns']:
                    for s in slices:
                        ref_s = DataSlice.from_dataframe(sfp.df_export[~sfp.df_export.index.isin(s)])
                        y = sfp.df_export[c][s]
                        ref_y = sfp.df_export[c][ref_s]
                        s, ref_s = statistic.metric_function(s, ref_s, y, ref_y, c)

        # update dfs
        for d, slices in sfp.slices.items():
            description_, size_, metric_values_ = sfp.extract_slice_info(slices)

            sfp.data = dict(
                description=description_,
                size=size_,
                metric_values=metric_values_
            )
            df_slices = sfp.finalize_metric_processing()
            df_slices.to_csv(sfp.data_output_dir / f'slices_degree{d}_fairness.csv')

            # Only add new columns, do not overwrite dataframe
            new_columns = list(c for c in df_slices.columns if c not in sfp.dfs[d].columns)
            for c in new_columns:
                sfp.dfs[d][c] = df_slices[c]

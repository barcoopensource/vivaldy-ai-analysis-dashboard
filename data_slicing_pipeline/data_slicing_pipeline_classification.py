from data_slicing_pipeline.data_slicing_pipeline_abstract import DataSlicingPipelineAbstract
from metrics.evaluation.metrics_evaluation_binary_classification import MetricsEvaluationBinaryClassification, MetricsEvaluationBinaryClassificationMultithreshold
from data_slicing.data_slicing import DataSlice

from types import FunctionType
from types import ModuleType
from typing import List
import pandas as pd
from pathlib import Path


class DataSlicingPipelineClassification(DataSlicingPipelineAbstract):
    def __init__(self, data_output_dir: Path, degrees: int, settings_module: ModuleType):
        super().__init__(data_output_dir, degrees, settings_module)
        formatting_settings = self.settings.get('dataframe').get('formatting').get('binary_classification')
        metric_settings = self.settings.get('evaluation').get('binary_classification').get('metrics')

        self.PRED_LABEL_INPUT = formatting_settings.get('pred_label_input')
        self.GT_LABEL_INPUT = formatting_settings.get('gt_label_input')

        self.PRED_LABEL_OUTPUT = formatting_settings.get('pred_label_output')
        self.GT_LABEL_OUTPUT = formatting_settings.get('gt_label_output')

        self.NA_STRICT = formatting_settings.get('na_strict')
        self.NA_FILL = formatting_settings.get('na_fill')

        self.ADDITIONAL_COLUMNS_TO_EXPORT = self.settings.get('dataframe').get('export')

        self.META_DATA_FIELDS_OF_INTEREST = self.settings.get(
            'slicing').get('meta_data_fields_of_interest')

        self.CLASSIFICATION_THRESHOLD = formatting_settings.get('classification_threshold')

    def prepare_dataframe(self, df: pd.DataFrame, filters: List[FunctionType] = []):
        # Optionally, prefilter the dataframe
        for f in filters:
            df = df[f(df)]
        assert df.index.name == self.DATA_IDX, 'Dataframe does not have the expected index {DATA_IDX}. Please correct this in the preprocesing step and/or settings.'

        # Create copy of the dataframe to be exported
        df_export = df.copy(deep=True)

        # Select fields to consider
        meta_data_fields_of_interest = self.META_DATA_FIELDS_OF_INTEREST

        # Drop and rename possible conflicting fields
        for k, v in meta_data_fields_of_interest.items():
            if (k != v) and (v in df.columns):
                df = df.drop(columns=[v])
            if (k != v) and (v in df_export.columns):
                df_export = df_export.drop(columns=[v])

        df = df.rename(columns={self.GT_LABEL_INPUT: self.GT_LABEL_OUTPUT,
                                self.PRED_LABEL_INPUT: self.PRED_LABEL_OUTPUT})
        df = df.rename(columns=meta_data_fields_of_interest)
        df = df[list(meta_data_fields_of_interest.values(
        )) + [self.GT_LABEL_OUTPUT] + [self.PRED_LABEL_OUTPUT]]
        df_export = df_export.rename(columns=meta_data_fields_of_interest)

        # Make sure no NaNs are present
        if self.NA_STRICT:
            # NA values are not overwritten, and the full dataframe is asserted to have no NA values
            is_NaN = df.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = df[row_has_NaN]
            assert rows_with_NaN.empty
        else:
            df.fillna(self.NA_FILL, inplace=True)
            df_export.fillna(
                self.NA_FILL, inplace=True)

        # NOTE: Target column should be last
        # extract gt
        df = df[[col for col in df.columns if col != self.PRED_LABEL_OUTPUT] + [self.PRED_LABEL_OUTPUT]]
        X, y_gt, y_pred = df[df.columns.difference([self.PRED_LABEL_OUTPUT, self.GT_LABEL_OUTPUT])], \
            df[self.GT_LABEL_OUTPUT], \
            df[self.PRED_LABEL_OUTPUT]

        # Strip df_export dataframe from unnecessary columns
        columns_to_keep = list(meta_data_fields_of_interest.values()) + \
            [self.PRED_LABEL_OUTPUT, self.GT_LABEL_OUTPUT] + \
            self.ADDITIONAL_COLUMNS_TO_EXPORT

        columns_to_drop = []
        for c in df_export.columns:
            if c not in columns_to_keep:
                columns_to_drop.append(c)
        df_export.drop(columns=columns_to_drop, inplace=True)

        df_export[self.GT_LABEL_OUTPUT] = df[self.GT_LABEL_OUTPUT]
        df_export[self.PRED_LABEL_OUTPUT] = df[self.PRED_LABEL_OUTPUT]

        return X, y_gt, y_pred, df_export

    def evaluate_metrics(self, slices: List[DataSlice], mp_settings={}) -> List[DataSlice]:
        # Binary classification metrics
        classification_threshold = self.CLASSIFICATION_THRESHOLD
        Y = [(self.y_gt[s], (self.y_predict[s] >= classification_threshold).astype(int)) for s in slices]
        mbc = MetricsEvaluationBinaryClassification()
        slices = mbc.process(slices, Y, settings=self.settings, mp_settings=mp_settings)

        # Multithreshold metrics
        Y = [(self.y_gt[s], self.y_predict[s]) for s in slices]
        mbcm = MetricsEvaluationBinaryClassificationMultithreshold()
        slices = mbcm.process(slices, Y, settings=self.settings, mp_settings=mp_settings)

        # Group comparison metrics for binary classification
        if 'metrics_group_comparison' in self.settings.get('evaluation').get('binary_classification'):
            from metrics.evaluation.metrics_evaluation_binary_classification_group_comparison import MetricsEvaluationBinaryClassificationGroupComparison
            mbcgc = MetricsEvaluationBinaryClassificationGroupComparison()
            ref_slices = mbcgc.get_complementary_slices(slices, self.y_predict)
            Y = [(self.y_gt[s], (self.y_predict[s] >= classification_threshold).astype(int)) for s in slices]
            ref_Y = [(self.y_gt[s], (self.y_predict[s] >= classification_threshold).astype(int)) for s in ref_slices]
            slices, _ = mbcgc.process(slices, ref_slices, Y, ref_Y, settings=self.settings, mp_settings=mp_settings)

        return slices

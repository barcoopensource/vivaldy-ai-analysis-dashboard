from types import FunctionType
from types import ModuleType
import pandas as pd
import numpy as np

from data_slicing.data_slicing import DataSlice, DataSlicing
from typing import List
import time
from pathlib import Path


class DataSlicingPipelineAbstract:
    settings: ModuleType

    def __init__(self, data_output_dir: Path, degrees: int, settings_module: ModuleType):
        self.X = None
        self.y_gt = None
        self.y_predict = None
        self.meta_data_labels = {}
        self.meta_data = None
        self.data = None
        self.data_output_dir = data_output_dir
        self.degrees = degrees

        self.settings_module = settings_module
        self.settings = settings_module.settings
        self.DATA_IDX = self.settings.get('dataframe').get('index')

    """Inspired by https://github.com/yeounoh/slicefinder"""
    def preprocessing(self, y, data):
        self.y_predict = y
        self.X, self.y_gt = data

        # Encode categorical features
        meta_data_labels = {}
        for c in self.X.columns:
            meta_data_labels[c] = {}
            for new_label, category in enumerate(self.X[c].unique()):
                meta_data_labels[c][category] = new_label
            self.X[c] = np.vectorize(meta_data_labels[c].__getitem__)(self.X[c])
        self.meta_data_labels = meta_data_labels

    def prepare_dataframe(df: pd.DataFrame, filters: List[FunctionType] = []):
        """This should be implemented in the child class"""
        pass

    def process(self, df: pd.DataFrame, stop_after_df_preparation: bool = False):
        df.index = df.index.astype(str, copy=False)
        self.df_original = df
        X, y_gt, y_pred, df_export = self.prepare_dataframe(df)
        self.df_export = df_export

        for k, v in self.settings.get('evaluation').get('additional_metrics').items():
            k.metric_function(self, **v)

        self.data_output_dir.mkdir(exist_ok=True, parents=True)
        df_export.to_csv(self.data_output_dir / 'df.csv', date_format="%Y-%m-%dT%H:%M:%S.%fZ")
        if stop_after_df_preparation:
            print(f"Data preparation finished for {self.data_output_dir.name}")
            return

        self.dfs = {}
        self.slices = {}

        self.preprocessing(y_pred, (X, y_gt))
        for d in self.degrees:
            # Find slices
            print(f'Slicing meta data labels for degree {d}')
            start = time.time()
            if d > 0:
                mp_settings = self.settings.get('mp').get(d).get('mp_data_slicing')
                minimum_samples  = self.settings.get('slicing').get('data_slicing_minimum_samples')
                slices = self.find_slices(d, minimum_samples, mp_settings)
            else:
                s_d0 = DataSlice.from_dataframe(X)
                slices = [s_d0]
            # Store slices for potental further usage during post-processing steps
            self.slices[d] = slices

            print(f"Slicing for degree {d} finished in {time.time() - start} s")
            start_metric_evaluation = time.time()

            # Evaluate metrics
            mp_settings = self.settings.get('mp').get(d).get('mp_metrics')
            self.get_metric_values(slices, mp_settings)
            df_slices = self.finalize_metric_processing()

            # Export metric results for degree d
            df_slices.to_csv(self.data_output_dir / f'slices_degree{d}_metrics.csv')
            print(f"Metric evaluation for degree {d} finished in {time.time() - start_metric_evaluation} s")
            print(f"Degree {d} finished in {time.time() - start} s")
            print("")
            self.dfs[d] = df_slices

        # Execute optional postprocessing steps
        print('Executing chosen postprocessing steps')
        for postprocessing_cls, postprocessing_cls_args in self.settings.get('evaluation').get('postprocessing').items():
            print(f'Postprocessing step: {postprocessing_cls.__name__}')
            postprocessing_cls.process(self, postprocessing_cls_args)

        # Export
        print('Exporting data')
        for d in self.degrees:
            self.export_df(self.dfs[d], f'slices_degree{d}.csv')

        print('Data slicing finished')
        print('')

    def find_slices(self, chosen_degree, minimum_samples, mp_settings):
        sf = DataSlicing(self.y_predict, (self.X, self.y_gt), minimum_samples, mp_settings)
        slices = sf.find_slice(degree=chosen_degree)
        return slices

    def get_metric_values(self, slices, mp_settings):
        if len(slices) > 0:
            slices = self.evaluate_metrics(slices, mp_settings)
        description_, size_, metric_values_ = self.extract_slice_info(slices)

        self.data = dict(
            description=description_,
            size=size_,
            metric_values=metric_values_
        )

    def export_df(self, df, df_name):
        df.to_csv(self.data_output_dir / df_name)

    def evaluate_metrics(self, *args, mp_settings={}, **kwargs):
        """This should be implemented in the child class"""
        pass

    def extract_slice_info(self, slices: List[DataSlice]):
        from itertools import chain
        description_ = list()
        size_ = list()
        metric_values_ = {}

        # Enable filling of non-evaluated metrics by listing all used metrics upfront:
        unique_metrics = set(chain(*[s.get_all_metrics().keys() for s in slices]))
        for metric in unique_metrics:
            metric_values_[metric] = []

        # inverse meta_data_labels lookup
        inverse_meta_data_labels = {}
        for category in self.meta_data_labels.keys():
            inverse_meta_data_labels[category] = {v: k for k, v in self.meta_data_labels[category].items()}

        description_separator : str = self.settings.get('dataframe').get('formatting').get('description_separator')

        for s in slices:
            description_elements = []
            for k, v in list(s.attrs['slice_info'].items()):
                category_value = inverse_meta_data_labels[k][v]
                description_elements.append(str(k) + ':' + str(category_value))
            description_.append(description_separator.join(description_elements))
            size_.append(s.size)
            for metric in unique_metrics:
                value = s.get_all_metrics().get(metric, np.nan)
                metric_values_[metric].append(value)
            assert len(description_) == len(size_)
            for k, v in metric_values_.items():
                assert len(v) == len(size_)
        return description_, size_, metric_values_

    def finalize_metric_processing(self) -> pd.DataFrame:
        """Extract optional additional metrics and cleanup dataframe"""
        description_separator : str = self.settings.get('dataframe').get('formatting').get('description_separator')
        # Extract additional metric values
        for metric, values in self.data['metric_values'].items():
            self.data[metric] = values
        self.data.pop('metric_values')
        df = pd.DataFrame.from_dict(self.data)

        fields = [s.split(description_separator) for s in df['description'].to_numpy() if (len(df['description']) > 1)]
        for idx in range(len(fields)):
            for field in fields[idx]:
                k, v = field.split(':')
                df.loc[idx, k] = v

        # Cleanup
        df[df.isin([np.inf, -np.inf])] = np.nan
        return df

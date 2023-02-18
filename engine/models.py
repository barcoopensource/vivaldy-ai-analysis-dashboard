# import engine.utils.utils as utils
import pandas as pd
import operator
from collections.abc import Callable
from typing import Union
from dataclasses import dataclass


@dataclass
class ModelData:
    name: str
    slice_df: pd.DataFrame
    degree: int = None
    slice_label_value: str = None
    ranking_method: str = None
    slice_df_filtered_degree_slice: pd.DataFrame = None
    slice_df_filtered: pd.DataFrame = None


class ModelFilter:
    def filter_degree_slices(self, degree: int, slice_label_value: str, ranking_method: str) -> None:
        update = self.degree != degree or self.slice_label_value != slice_label_value or self.ranking_method != ranking_method
        if update:
            slice_labels = slice_label_value.split(' & ')
            slice_df = self.slice_df.copy(deep=True)
            slice_df = slice_df[slice_df['degree'] == degree]
            for c in slice_labels:
                slice_df = slice_df[~slice_df[c].isna()]
            self.slice_df_filtered_degree_slice = slice_df
            self.degree = degree
            self.slice_label_value = slice_label_value

        self.slice_df_filtered = self.slice_df_filtered_degree_slice
        return update

    def filter_threshold(self, column: str, threshold_value: float, op: Callable[[any, any], bool] = operator.ge) -> None:
        self.slice_df_filtered = self.slice_df_filtered[op(self.slice_df_filtered[column], threshold_value)]

    def filter_contains_values(self, column: str, options: list[Union[str, float, int]]) -> None:
        self.slice_df_filtered = self.slice_df_filtered[self.slice_df_filtered[column].isin(options)]

    def sort(self, columns: list[str], ascending: bool = False) -> None:
        self.slice_df_filtered = self.slice_df_filtered.sort_values(by=columns, ascending=ascending)


class Model(ModelData, ModelFilter):
    """This class holds the name and (optionally) dataframes of a model with functionality to filter."""
    def __init__(self, name: str, load=False) -> None:
        if load:
            self.__parse_contents_from_model_selection(name)
        else:
            self.slice_df = None
            self.full_df = None
        self.name = name

    def set_highlighting(self, settings, ranking_method: str) -> None:
        self.slice_df['highlight'] = False
        if ranking_method is None:
            return
        # Obtain sorting column
        sort_column = f'{settings.SORT_ID}{ranking_method}'
        if sort_column not in self.slice_df.columns:
            sort_column = settings.DEFAULT_SORT_COLUMN

        idx = pd.Index([])
        for degree in self.slice_df['degree'].unique():
            df_sorted = self.slice_df[self.slice_df['degree'] == degree]

            # Sort and extract index for top N rows
            df_sorted = df_sorted.sort_values(sort_column, inplace=False)
            df_sorted = df_sorted[~df_sorted[sort_column].isna()]

            if len(df_sorted) >= settings.HIGHLIGHT_N_TOP_RANKED:
                top_N_index = df_sorted.iloc[0:settings.HIGHLIGHT_N_TOP_RANKED].index
            else:
                top_N_index = df_sorted.index
            idx = idx.append(top_N_index)

        # set highlighting to True for the found indices within slice_df, i.e. highlighting should be done before further filtering
        self.slice_df.loc[self.slice_df.index.isin(idx), 'highlight'] = True

    def __parse_contents_from_model_selection(self, model: str) -> None:
        import dashboard_settings as dashboard_settings
        from pathlib import Path
        dfs = []
        for d in range(0, dashboard_settings.NB_DEGREES + 1):
            try:
                model_path: Path = dashboard_settings.MODEL_ROOT_FOLDER / model
                csv_file_name: Path = model_path / \
                    f'slices_degree{d}{dashboard_settings.df_file_suffix}.csv'
                df_ = pd.read_csv(csv_file_name, low_memory=False, index_col=0)
                df_['degree'] = d
                dfs.append(df_)

            except Exception as e:
                print(f'Failed to load csv for degree {d}')
                print(e)
        df = pd.concat(dfs)
        df = df.reset_index(drop=True)
        full_df = pd.read_csv(
            dashboard_settings.MODEL_ROOT_FOLDER / model / 'df.csv', low_memory=False)

        self.slice_df = df
        self.full_df = full_df

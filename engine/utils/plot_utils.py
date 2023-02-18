from __future__ import annotations
import itertools
import pandas
from engine.properties import EngineProperties
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine.models import Model
    import numpy as np

dashboard_settings = EngineProperties.settings


def add_metric_reference(model: Model, degree: int, slice_label_value: str, metric: str) -> pandas.DataFrame:
    """Return prepared dataframe of the model according to the metric type."""
    from functools import partial
    filtered_df = model.slice_df_filtered.copy()
    metric_column_name = dashboard_settings.METRIC_OPTIONS[metric]['column_name']
    if not filtered_df.empty:
        if metric not in dashboard_settings.BOX_PLOT_METRICS:
            # Use bar plot visualization for metric value
            filtered_df['x'] = filtered_df["description"]
            filtered_df['y'] = filtered_df[dashboard_settings.METRIC_OPTIONS[metric]
                                           ['column_name']]
            filtered_df['LB'] = 0
            filtered_df['UB'] = 0

            if (f'{metric_column_name}_LB' in filtered_df.columns) and (f'{metric_column_name}_UB' in filtered_df.columns):
                filtered_df['LB'] = filtered_df['y'] - \
                    filtered_df[f'{metric_column_name}_LB']
                filtered_df['UB'] = filtered_df[f'{metric_column_name}_UB'] - \
                    filtered_df['y']

            slice_labels = [None] if degree == 1 else list(
                [s.split(':')[0] for s in filtered_df.iloc[0]['description'].split('  ')])
            parent_degree = 0 if degree == 1 else 1

            for d in range(degree):
                filtered_df[f'reference_option{d}'] = filtered_df.apply(
                    lambda row: extract_reference_values(model.slice_df, row, slice_labels[d], parent_degree, metric_column_name), axis=1)
                filtered_df[f'reference_option{d}_LB'] = filtered_df.apply(
                    lambda row: extract_reference_values(model.slice_df, row, slice_labels[d], parent_degree, metric_column_name, column_suffix='_LB'), axis=1)
                filtered_df[f'reference_option{d}_UB'] = filtered_df.apply(
                    lambda row: extract_reference_values(model.slice_df, row, slice_labels[d], parent_degree, metric_column_name, column_suffix='_UB'), axis=1)
        else:
            slice_labels = slice_label_value.split(' & ')
            filtered_df = filtered_df.apply(partial(
                add_xy_data, metric=metric, column_values=slice_labels, full_df=model.full_df), axis=1)
            filtered_df = pandas.concat(list(filtered_df))

    return filtered_df


def align_axis(sort_column: str, column_values: list[str], df: pandas.DataFrame, column_out: str = "y") -> np.array:
    """Make the order of the dataframe match the column_values in sort_column column. Return the column_out column.
    """
    import numpy as np
    res = np.array([])
    for value in column_values:
        rows_df = df[df[sort_column] == value]
        if rows_df.empty:
            res = np.append(res, np.array([np.nan]))
        else:
            res = np.append(res, rows_df[column_out])
    return res


def extract_reference_values(slice_df: pandas.DataFrame,
                             row: pandas.Series,
                             description_parent_degree: str,
                             parent_degree: str,
                             metric_column_name: str,
                             column_suffix: str = '') -> float:
    import numpy
    try:
        sel = slice_df['degree'] == parent_degree
        if description_parent_degree is not None:
            slice_label_value = row[description_parent_degree]
            sel &= slice_df[description_parent_degree] == slice_label_value
        column_name = f'{metric_column_name}{column_suffix}'
        if column_name in slice_df.columns:
            reference_value = slice_df.loc[sel, column_name].iloc[0]
        else:
            reference_value = numpy.nan
    except IndexError:
        reference_value = numpy.nan
    return reference_value


def add_xy_data(row: pandas.Series, metric: str, column_values: list[str], full_df: pandas.DataFrame) -> pandas.DataFrame:
    x = row['description']
    full_df_filtered = full_df.copy(deep=True)
    df = pandas.DataFrame()
    for c in column_values:
        full_df_filtered = full_df_filtered[full_df_filtered[c] == row[c]]
    for c in column_values:
        df[c] = full_df_filtered[c]
    df['x'] = x
    df['y'] = full_df_filtered[dashboard_settings.METRIC_OPTIONS[metric]['column_name']]
    return df


def calculate_y_values(n: int) -> list[int]:
    res = []
    for i in range(n):
        res.append(1 / (2 * n) + i / n)
    return res


def get_plotted_value_range(df, metric):
    if 'ylim' in dashboard_settings.METRIC_OPTIONS[metric].keys():
        return dashboard_settings.METRIC_OPTIONS[metric]['ylim']
    else:
        return min([0, df['y'].min()]), max([0, df['y'].max()])


def prepare_plot_kwargs(n_models, n_rows=1, is_violin_or_box_plot=False) -> list(dict):
    plot_kwargs = []
    for x, y in itertools.product(range(n_rows), range(n_models)):
        plot_kwargs.append({})

    # Violin plot control
    if is_violin_or_box_plot:
        if n_models == 2:
            for row_idx in range(n_rows):
                plot_kwargs[0 + row_idx * n_models]['side'] = 'negative'
                plot_kwargs[1 + row_idx * n_models]['side'] = 'positive'
        else:
            for w in plot_kwargs:
                w['side'] = 'both'

        if n_models > 2:
            for row_idx, model_idx in itertools.product(range(n_rows), range(n_models)):
                plot_kwargs[row_idx * n_models + model_idx]['box_plot'] = True

    # Legend control
    if not is_violin_or_box_plot:
        if n_models == 1:
            plot_kwargs[0]['showlegend'] = True
        elif n_models > 1:
            plot_kwargs[0]['showlegend'] = True
            for w in plot_kwargs[1:]:
                w['showlegend'] = False
    else:
        for w in plot_kwargs[:]:
            w['showlegend'] = False

    # Show Reference CI control
    for x, y in itertools.product(range(n_rows), range(n_models)):
        plot_kwargs[x * n_models + y]['show_reference_CI'] = False
        if y == 0:
            plot_kwargs[x * n_models + y]['show_reference_CI'] = True
    return plot_kwargs


def is_violin_or_box_plot(metric):
    return metric in dashboard_settings.BOX_PLOT_METRICS

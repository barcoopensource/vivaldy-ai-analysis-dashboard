from __future__ import annotations
from copy import deepcopy
from functools import reduce
import pandas
import plotly.graph_objects as go
from engine.utils.plot_utils import add_metric_reference, align_axis, calculate_y_values, get_plotted_value_range, prepare_plot_kwargs, is_violin_or_box_plot
from engine.properties import EngineProperties
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine.models import Model

settings = EngineProperties.settings


def generate_bar_graph(models: list[Model],
                       column: str,
                       degree: int,
                       slice_label_value: str,
                       model_colors: list[str],
                       array_graph: bool = True,
                       axis_order: list[str] = [],
                       detailled=False) -> go.Figure:
    """Generate plot independent from metric, e.g. for samples per slice."""
    if len(models) == 0:
        return None
    y_detailled = None
    filtered_dfs = [model.slice_df_filtered.copy() for model in models]
    for df in filtered_dfs:
        df["x"] = df['description']
        df["y"] = df[column]

    if column == 'size' and detailled:
        if all([col in settings.DETAILLED_BAR_PLOT_ORDER for col in ["tn", "tp", "fn", "fp"]]):
            y_detailled = settings.DETAILLED_BAR_PLOT_ORDER
        else:
            y_detailled = ["tn", "tp", "fn", "fp"]

    if array_graph:
        fig = create_bar_array_figure(
            filtered_dfs, degree, slice_label_value, model_colors, None, axis_order, y_detailled=y_detailled)
    else:
        fig = create_bar_line_figure(
            filtered_dfs, degree, model_colors, None, y_detailled=y_detailled)
    return fig


def generate_metric_graph(models: list[Model],
                          degree: int,
                          slice_label_value: str,
                          metric: str,
                          model_colors: list[str],
                          array_graph: bool = True,
                          axis_order: list[str] = []) -> go.Figure:
    """Generate metric plot."""
    plot_dfs = []
    if len(models) == 0:
        return None

    for model in models:
        plot_dfs.append(model.slice_df_filtered)

    if array_graph:
        fig = create_bar_array_figure(
            plot_dfs, degree, slice_label_value, model_colors, metric, axis_order)
    else:
        fig = create_bar_line_figure(plot_dfs, degree, model_colors, metric)

    return fig


def add_reference_CI(fig: go.Figure,
                     df: pandas.DataFrame,
                     degree: int,
                     x: np.array,
                     sort_column: str,
                     column_values: str,
                     showlegend: bool,
                     row: int,
                     col: int) -> go.Figure:
    """Add CIs to the given figure"""
    if not df.empty:
        shard_descriptions = list(
            [s.split(':')[0] for s in df.iloc[0]['x'].split('  ')])

        for d in range(degree):
            fig.add_trace(go.Scatter(x=x - 0.1 + d * 0.2,
                                     y=align_axis(
                                         sort_column, column_values, df, f'reference_option{d}'),
                                     mode='markers',
                                     name='D0' if degree == 1 else f'{shard_descriptions[d]} D1',
                                     error_y=dict(type='data',
                                                  symmetric=False,
                                                  array=df[f'reference_option{d}_UB'] - df[f'reference_option{d}'],
                                                  arrayminus=df[f'reference_option{d}'] - df[f'reference_option{d}_LB'],
                                                  color=settings.REFERENCE_SHARD_COLOR[d]
                                                  ),
                                     showlegend=showlegend,
                                     marker=dict(
                                         color=[settings.REFERENCE_SHARD_COLOR[d]] * len(x)),
                                     ),
                          row=row,
                          col=col)
    return fig


def add_plot(fig: go.Figure,
             df: pandas.DataFrame,
             degree: int,
             x: np.array,
             sort_column: str,
             column_values: str,
             column_out: str = "y",
             color: str = None,
             metric: str = None,
             show_reference_CI: bool = False,
             y_detailled: list[str] = None,
             showlegend: bool = False,
             row: int = None,
             col: int = None,
             ticktext: list[str] = None,
             title_text: str = None,
             side="both",
             box_plot: bool = False,
             model_id: int = None) -> go.Figure:
    """Add bars/violins to the given row/column based on wether there is a metric and wether the metric is a box plot metric or not.
    """
    if metric is not None:
        if is_violin_or_box_plot(metric):
            return add_violin(fig, df, x, sort_column, column_values, column_out, color, showlegend, row, col, ticktext, title_text, side, box_plot, model_id)
    return add_bar(fig, df, degree, x, sort_column, column_values, column_out, color, metric, show_reference_CI, y_detailled, showlegend, row, col, ticktext, title_text, model_id)


def add_violin(fig: go.Figure,
               df: pandas.DataFrame,
               x: np.array,
               sort_column: str,
               column_values: list[str],
               column_out: str,
               color: str,
               showlegend: bool,
               row, col,
               ticktext: str,
               title_text: str,
               side: str,
               box_plot: bool,
               model_id: str = None) -> go.Figure:
    """Add violins to the given row/col"""
    if not box_plot:
        x = []
        for x_value in column_values:
            length = len(df[df[sort_column] == x_value])
            x_value = str(x_value)
            if length == 0:
                x += [x_value]
            else:
                x += [x_value] * length
        y = align_axis(sort_column, column_values, df, column_out)
        fig.add_trace(
            go.Violin(
                x=x,
                y=y,
                legendgroup='No',
                scalegroup='No',
                marker=dict(color=color),
                showlegend=showlegend,
                side=side),
            row=row,
            col=col)

        fig.update_xaxes(
            title_text=title_text,
            # tickmode='array',
            # tickvals=x,
            # ticktext=ticktext if ticktext is not None else column_values,
            row=row,
            col=col
        )
        fig.update_layout(boxmode='group')
    else:
        for x_value in column_values:
            y = df[df[sort_column] == x_value][column_out]
            x_ = [str(x_value)] * len(y)
            fig.add_trace(
                go.Box(
                    y=y,
                    x=x_,
                    marker=dict(color=color),
                    showlegend=showlegend,
                    name=str(x_value),
                    # name=str(model_id + 1)
                    alignmentgroup=str(x_value)
                ),
                row=row,
                col=col,
            )
        fig.update_layout(boxmode='group')
    return fig


def add_bar(fig: go.Figure,
            df: pandas.DataFrame,
            degree: int,
            x: np.array,
            sort_column: str,
            column_values: str,
            column_out: str = "y",
            color: str = None,  # "blue",
            metric: str = None,
            show_reference_CI: bool = False,
            y_detailled: list[str] = None,
            showlegend: bool = False,
            row: int = None,
            col: int = None,
            ticktext: list[str] = None,
            title_text: str = None,
            model_id: str = None) -> go.Figure:
    """Add bars to the given row/col"""
    if y_detailled is not None and metric is None:
        for idx, column in enumerate(y_detailled):
            if model_id is not None:
                x_ = [column_values, [str(model_id + 1)] * len(column_values)]
            else:
                x_ = column_values
            fig.add_trace(go.Bar(
                x=x_,
                y=align_axis(sort_column, column_values, df, column),
                name=column,
                marker=dict(color=[settings.Y_DETAILLED_COLORS[idx % len(
                    settings.Y_DETAILLED_COLORS)]] * len(x)),
                showlegend=showlegend
            ),
                row=row,
                col=col,
            )
        fig.update_layout(barmode='stack')
        fig.update_layout(xaxis_title=title_text)
    else:
        if metric is not None:
            error_y = {
                "type": "data",
                "symmetric": False,
                "array": align_axis(sort_column, column_values, df, 'UB'),
                "arrayminus": align_axis(sort_column, column_values, df, 'LB')
            }
        else:
            error_y = None
        # Add bar
        fig.add_trace(
            go.Bar(
                x=x,
                y=align_axis(sort_column, column_values, df, column_out),
                marker=dict(color=[color] * len(x)),
                error_y=error_y,
                name=metric,
                showlegend=showlegend
            ),
            row=row,
            col=col,
        )

        # Add highlighting
        if 'highlight' in df.columns and metric is not None:
            highlighting_y0, highlighting_y1 = get_plotted_value_range(df, metric)
            df_highlight = df[df['highlight']]
            for x_highlight in df_highlight.index:
                x_pos = list(align_axis(sort_column, column_values, df.reset_index(drop=False), 'index').astype(int)).index(x_highlight)
                fig.add_shape(type='rect',
                              x0=x_pos - 0.5,
                              y0=highlighting_y0,
                              x1=x_pos + 0.5,
                              y1=highlighting_y1,
                              xref='x',
                              yref='y',
                              fillcolor='lightgoldenrodyellow',
                              layer='below',
                              line_width=0,
                              row=row,
                              col=col
                              )
        if metric is not None and show_reference_CI:
            fig = add_reference_CI(
                fig, df, degree, x, sort_column, column_values, showlegend, row, col)
        fig.update_xaxes(
            tickmode='array',
            tickvals=x,
            ticktext=ticktext if ticktext is not None else column_values,
            title_text=title_text,
            row=row,
            col=col
        )
    return fig


def create_bar_line_figure(dfs: list[pandas.DataFrame], degree: int, model_colors: list[str], metric: str = None, y_detailled: list[str] = None) -> go.Figure:
    """Create plot with all labels on the x axis."""
    descriptions = dfs[0]['x'].unique()  # No additional sorting, was done in the Widget
    x_numeric = np.array(range(len(descriptions)))

    ticktext = []
    labels = []

    first_description = True
    for description in descriptions:
        label_values = description.split(' ')
        tick = ""
        for label_value in label_values:
            if label_value != "":
                if first_description:
                    labels.append(label_value.split(':')[0])
                tick += label_value.split(':')[-1]
                tick += " & "
        first_description = False
        ticktext.append(tick[:-3])

    fig = go.Figure()
    plot_kwargs = prepare_plot_kwargs(len(dfs), 1, is_violin_or_box_plot(metric))

    for idx, (df, w) in enumerate(zip(dfs, plot_kwargs)):
        fig = add_plot(
            fig=fig,
            df=df,
            degree=degree,
            x=x_numeric,
            sort_column="x",
            column_values=descriptions,
            color=model_colors[idx % len(model_colors)],
            metric=metric,
            y_detailled=y_detailled,
            ticktext=ticktext,
            title_text=' & '.join(labels),
            model_id=idx if len(dfs) > 1 else None,
            **w
        )

    if metric is not None and y_detailled is None:
        if 'ylim' in settings.METRIC_OPTIONS[metric].keys():
            fig.update_yaxes(
                range=settings.METRIC_OPTIONS[metric]['ylim'], title_text='')

    if len(ticktext) > 11:
        width = (80 * len(ticktext))
    else:
        width = 900
    fig.update_layout(height=(500), width=width, margin={
                      'l': 125, 'r': 125, 't': 20, 'b': 100})
    return fig


def create_bar_array_figure(dfs: list[pandas.DataFrame],
                            degree: int,
                            slice_label_value: str,
                            model_colors: list[str],
                            metric: str = None,
                            axis_order: list[str] = [],
                            y_detailled: list[str] = None) -> go.Figure:
    """Create plot with only one label on the x axis and the others on the y axis."""
    from plotly.subplots import make_subplots

    reference_df = dfs[0]
    axes_ordered = []

    # enable predefined sorting when degree is set to 1
    if degree == 1:
        def local_sort(x):
            return x
    else:
        local_sort = sorted

    for label in axis_order:
        axes_ordered.append((label, local_sort(reference_df[label].unique())))
    if len(axis_order) != degree:
        axes_unordered = {}
        slice_labels = slice_label_value.split(" & ")
        for label in slice_labels:
            if label not in axis_order:
                axes_unordered[label] = local_sort(reference_df[label].unique())

        while axes_unordered != {}:
            max_key, max_value = max(
                axes_unordered.items(), key=lambda x: len(x[1]))
            axes_ordered.append((max_key, list(max_value)))
            axes_unordered.pop(max_key)

    # explicitly convert values in axes_ordered to str
    axes_ordered_lookup = deepcopy(axes_ordered)
    axes_ordered_text = list((str(k), list(map(str, v))) for (k, v) in axes_ordered)
    del axes_ordered  # Avoid using this variable. Use axes_ordered_lookup and axes_ordered_text

    if degree == 1:
        n_rows = 1
        fig = make_subplots(rows=n_rows, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.12 / n_rows)
        x_numeric = np.array(range(len(axes_ordered_text[0][1])))
        plot_kwargs = prepare_plot_kwargs(len(dfs), n_rows, is_violin_or_box_plot(metric))

        row_index = 0
        for idx, df in enumerate(dfs):
            w = plot_kwargs[row_index * len(dfs) + idx]
            row_df = df
            fig = add_plot(
                fig=fig,
                df=row_df,
                degree=degree,
                x=x_numeric,
                sort_column=axes_ordered_text[0][0],
                column_values=axes_ordered_lookup[0][1],
                color=model_colors[idx % len(model_colors)],
                metric=metric,
                y_detailled=y_detailled,
                row=row_index + 1,
                col=1,
                model_id=idx if len(dfs) > 1 else None,
                title_text=axes_ordered_text[0][0],
                # show_reference_CI=True,
                **w
            )
        if metric is not None and y_detailled is None:
            if 'ylim' in settings.METRIC_OPTIONS[metric].keys():
                fig.update_yaxes(
                    range=settings.METRIC_OPTIONS[metric]['ylim'], title_text='')

        if len(x_numeric) > 11:
            width = (80 * len(x_numeric))
        else:
            width = 900
        fig.update_layout(height=(500), width=width, margin={'l': 125, 'r': 125, 't': 20, 'b': 100})

    if degree == 2:
        n_rows = len(axes_ordered_text[1][1])
        fig = make_subplots(rows=n_rows, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.12 / n_rows)
        x_numeric = np.array(range(len(axes_ordered_text[0][1])))
        plot_kwargs = prepare_plot_kwargs(len(dfs), n_rows, is_violin_or_box_plot(metric))
        for row_index in range(n_rows):
            for idx, df in enumerate(dfs):
                w = plot_kwargs[row_index * len(dfs) + idx]
                row_df = df[df[axes_ordered_lookup[1][0]]
                            == axes_ordered_lookup[1][1][row_index]]
                fig = add_plot(
                    fig=fig,
                    df=row_df,
                    degree=degree,
                    x=x_numeric,
                    sort_column=axes_ordered_text[0][0],
                    column_values=axes_ordered_lookup[0][1],
                    color=model_colors[idx % len(model_colors)],
                    metric=metric,
                    y_detailled=y_detailled,
                    row=row_index + 1,
                    col=1,
                    model_id=idx if len(dfs) > 1 else None,
                    # show_reference_CI=True,
                    **w
                )
            if metric is not None and y_detailled is None:
                if 'ylim' in settings.METRIC_OPTIONS[metric]:
                    fig.update_yaxes(
                        range=settings.METRIC_OPTIONS[metric]['ylim'], row=row_index + 1, col=1)
            fig.update_yaxes(
                title_text=axes_ordered_text[1][1][row_index], row=row_index + 1, col=1)
        # Add category labels on major x-axis and the two y-axes
        fig.add_annotation(
            text=axes_ordered_text[1][0],
            **settings.LAYOUT_SETTINGS['array-view']['x_label_left']
        )
        fig.add_annotation(
            text=axes_ordered_text[0][0],
            **settings.LAYOUT_SETTINGS['array-view']['y_label_bottom']
        )
        fig.update_layout(height=(
            200 * len(axes_ordered_text[1][1]) + 300), margin={'l': 125, 'r': 125, 't': 20, 'b': 100})

    if degree == 3:
        n_rows_1 = len(axes_ordered_text[2][1])
        n_rows_2 = len(axes_ordered_text[1][1])
        n_rows = n_rows_1 * n_rows_2
        fig = make_subplots(rows=n_rows, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.12 / n_rows)

        shapes = []  # used for different background color
        bg = True  # alternate bg colors
        x_numeric = np.array(range(len(axes_ordered_text[0][1])))
        fig['layout'].update(
            height=(200 * n_rows + 200))
        row = n_rows
        plot_kwargs = prepare_plot_kwargs(len(dfs), n_rows_1 * n_rows_2, is_violin_or_box_plot(metric))
        for row_index in range(n_rows_1):
            bg = not bg
            if bg:
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=-1,
                        y0=1 - 1 / n_rows_1 * (row_index + 1),
                        x1=len(axes_ordered_text[0][1]),
                        y1=1 - 1 / n_rows_1 * row_index,
                        fillcolor="rgb(185,185,185)",
                        opacity=1,
                        layer="below",
                        line_width=0,
                    )
                )
            row_dfs = [df[df[axes_ordered_lookup[2][0]] == axes_ordered_lookup[2][1][row_index]] for df in dfs]
            row = row - n_rows_2
            for row_index2 in range(n_rows_2):
                for idx, df in enumerate(row_dfs):
                    w = plot_kwargs[(row_index2 * n_rows_1 + row_index) * len(dfs) + idx]
                    row_df2 = df[df[axes_ordered_lookup[1][0]] == axes_ordered_lookup[1][1][row_index2]]
                    fig = add_plot(
                        fig=fig,
                        df=row_df2,
                        degree=degree,
                        x=x_numeric,
                        sort_column=axes_ordered_text[0][0],
                        column_values=axes_ordered_lookup[0][1],
                        color=model_colors[idx % len(model_colors)],
                        metric=metric,
                        y_detailled=y_detailled,
                        row=row + row_index2 + 1,
                        col=1,
                        model_id=idx if len(row_dfs) > 1 else None,
                        **w
                    )

                fig.update_yaxes(
                    title_text=axes_ordered_text[1][1][row_index2], row=row + row_index2 + 1, col=1)
                if metric is not None and y_detailled is None:
                    if 'ylim' in settings.METRIC_OPTIONS[metric]:
                        fig.update_yaxes(
                            range=settings.METRIC_OPTIONS[metric]['ylim'], row=row + row_index2 + 1, col=1)

            fig.add_annotation(
                text=axes_ordered_text[2][1][row_index],
                xref="paper", yref="paper",
                x=1.08,
                y=calculate_y_values(n_rows_1)[row_index],
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                textangle=90,
                font={'size': 14}
            )
        # Add category labels on major x-axis and the two y-axes
        fig.add_annotation(
            text=axes_ordered_text[2][0],
            **settings.LAYOUT_SETTINGS['array-view']['x_label_right']
        )
        fig.add_annotation(
            text=axes_ordered_text[1][0],
            **settings.LAYOUT_SETTINGS['array-view']['x_label_left']
        )
        fig.add_annotation(
            text=axes_ordered_text[0][0],
            **settings.LAYOUT_SETTINGS['array-view']['y_label_bottom']
        )
        # Preserve already drawn shapes
        for s in fig.layout.shapes:
            shapes.insert(0, s._props)

        fig.update_layout(
            shapes=shapes,
            paper_bgcolor='rgba(255,255,255,255)',
            plot_bgcolor='rgb(245,245,245)',
            margin={'l': 125, 'r': 125, 't': 20, 'b': 125}
        )
    return fig

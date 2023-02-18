from __future__ import annotations
from copy import deepcopy
from http.cookiejar import LoadError
from types import FunctionType
from typing import List
from dash import html, dcc, ctx
from dash.development.base_component import Component
from engine.models import Model
import engine.utils.utils as utils
import sys
from engine.widgets_abstract import Widget, Tab
import logging

from engine.utils.plot_utils import add_metric_reference


class PerformanceTab(Tab):
    def __init__(self) -> None:
        super().__init__("Performance per slice")

    def _calculate_output(self, inputs) -> any:
        # get inptus
        metric = inputs["metric-selector"]
        degree = inputs["degree-selector"]
        slice_label = inputs["slice-label-selector"]
        size_threshold = inputs["size-threshold"]
        label_value_filter = inputs["slice-label-value-filter"]
        sort_operator = inputs["sort-label-property-operator"]
        sort_columns = inputs["sort-label-property"]
        axis_order = utils.parse_axis_order(inputs["axis-label-selector"])
        graph_type = inputs["graph-type-selector"]
        sort_model_selector = inputs["sort-model-selector"]
        ranking_method = inputs["slice-label-ranking-selector"]

        model_colors = self.settings.Y_PERFORMANCE_COLORS

        loaded_models = utils.filter_and_sort_models(self.selected_models, inputs)

        if len(loaded_models) == 0:
            return []

        for m in loaded_models:
            m.set_highlighting(self.settings, ranking_method)

        for m in loaded_models:
            m.original_slice_df_filtered = deepcopy(m.slice_df_filtered)
            m.slice_df_filtered = add_metric_reference(
                m, degree, slice_label, metric)

        fig = utils.generate_metric_graph(
            loaded_models, degree, slice_label, metric, model_colors, graph_type == "array", axis_order)

        if fig is None:
            return []

        if "export-df-button" == ctx.triggered_id:
            utils.export_to_csvs(loaded_models, inputs, 'PerformanceTab')

        for m in loaded_models:
            m.slice_df_filtered = deepcopy(m.original_slice_df_filtered)
            del m.original_slice_df_filtered

        return [html.Div([
            html.H2(self.settings.METRIC_OPTIONS[metric]['display_name']),
            dcc.Graph(id='size_of_shards_histogram', figure=fig)
        ])]

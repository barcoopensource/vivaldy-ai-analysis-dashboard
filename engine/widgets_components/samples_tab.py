from __future__ import annotations
from types import FunctionType
from typing import List
from dash import html, dcc, ctx
from dash.development.base_component import Component
from engine.models import Model
import engine.utils.utils as utils
import sys
from engine.widgets_abstract import Widget, Tab
import logging


class SamplesTab(Tab):
    def __init__(self) -> None:
        super().__init__("Samples per slice")

    def _calculate_output(self, inputs) -> any:
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

        for m in loaded_models:
            m.set_highlighting(self.settings, ranking_method)

        fig = utils.generate_bar_graph(loaded_models, "size", degree, slice_label, model_colors,
                                       graph_type == "array", axis_order, detailled=self.settings.DETAILLED_BAR_PLOT)

        if fig is None:
            return []

        if "export-df-button" == ctx.triggered_id:
            utils.export_to_csvs(loaded_models, inputs, 'SamplesTab')

        return [html.Div([
            html.H2('Number of samples per slice:'),
            dcc.Graph(id='data_slicing_histogram', figure=fig)
        ])]

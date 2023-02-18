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


class MetricSelector(Widget):
    color_disabled = {'color': 'lightgrey'}
    color_enabled = {'color': 'Black'}

    def __init__(self) -> None:
        super().__init__("metric-selector",
                         {
                             "state_update": {
                                 "output": ["options", "value"],
                                 "input": ["value"]
                             },
                             "graph": {
                                 "input": ["value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        logging.info('MetricSelector get_html called')
        if self.selected_models[0] is None:
            for k, v in self.settings.METRIC_OPTIONS.items():
                v['disabled'] = True
                v['label'] = html.Span(
                    [v['legend']], style=self.color_disabled if v['disabled'] else self.color_enabled)
        return [
            html.H2('Metrics'),
            dcc.RadioItems(
                id=self.id,
                options=[{'value': k, 'label': v['label'], 'disabled': v['disabled']}
                         for k, v in self.settings.METRIC_OPTIONS.items()],
                value=self.settings.DEFAULT_METRIC),
        ]

    def _calculate_output(self, inputs) -> any:
        input_value = inputs[self.id]
        output = {'options': None, 'value': None}
        if self.selected_models[0] is None:
            for k, v in self.settings.METRIC_OPTIONS.items():
                v['disabled'] = True
                v['label'] = v['legend']
        else:
            slice_df_columns = utils.list_intersection(
                [model.slice_df.columns for model in self.selected_models if model is not None])
            full_df_columns = utils.list_intersection(
                [model.full_df.columns for model in self.selected_models if model is not None])
            for k, v in self.settings.METRIC_OPTIONS.items():
                metric_name = v['column_name']
                v['disabled'] = (metric_name not in slice_df_columns) if \
                    (metric_name not in self.settings.BOX_PLOT_METRICS) else \
                    (metric_name not in full_df_columns)
                v['label'] = html.Span(
                    [v['legend']], style=self.color_disabled if v['disabled'] else self.color_enabled)

        output['options'] = [{'value': k, 'label': v['label'], 'disabled': v['disabled']}
                             for k, v in self.settings.METRIC_OPTIONS.items()]
        if input_value is None:
            output['value'] = self.settings.RANKING_OPTIONS.get(
                inputs["slice-label-ranking-selector"], self.settings.DEFAULT_METRIC)
        else:
            output['value'] = input_value
        return output

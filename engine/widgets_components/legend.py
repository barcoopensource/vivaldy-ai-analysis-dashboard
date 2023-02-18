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


class Legend(Widget):
    def __init__(self) -> None:
        super().__init__("legend",
                         {
                             "state_update": {
                                 "output": ["children"] * self.settings.NB_MODELS
                             }
                         })
        self.nb_models = self.settings.NB_MODELS

    def get_html_components(self) -> list[Component]:
        return [html.H2('Legend')] + [html.Div([html.H3(f'Model {i+1}'), html.Div([html.Div(
            id=f'colour-block{i+1}')], id=f'legend-option{i+1}')], id=f'legend{i+1}') for i in range(self.nb_models)]

    def _calculate_output(self, inputs: dict[str: any]) -> tuple[list]:
        output = [[]] * self.nb_models
        for index, model in enumerate(self.selected_models):
            if model is not None:
                output[index] = [
                    html.Div([
                        html.H3(f'Model {index + 1}:'),
                        html.Div([
                            html.Div(id=f'colour-block{index + 1}', style={
                                'width': '15px',
                                'height': '15px',
                                'margin-right': '5px',
                                'border': '1px solid rgba(0, 0, 0, .2)',
                                'background-color': self.settings.Y_PERFORMANCE_COLORS[index % len(self.settings.Y_PERFORMANCE_COLORS)]#'#636EFB'
                                }),
                            html.Div(model.name)
                        ], id=f'legend-option{index + 1}', style={'display': 'flex'})
                    ])
                ]
        return tuple(output)

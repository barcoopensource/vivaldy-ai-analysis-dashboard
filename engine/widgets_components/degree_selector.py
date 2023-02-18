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


class DegreeSelector(Widget):
    def __init__(self) -> None:
        super().__init__(
            "degree-selector",
            {
                "state_update": {
                    "input": ["value"],
                },
                "graph": {
                    "input": ["value"]
                }
            })
        self.nb_degrees = self.settings.NB_DEGREES
        self.initial_selected_degree = self.settings.INITIAL_SELECTED_DEGREE

    def get_html_components(self) -> list[Component]:
        return [
            html.H2(['Slice level'], id='degree-selector-title'),
            dcc.RadioItems(
                id=self.id,
                options=[{'label': str(i + 1), 'value': i + 1}
                         for i in range(self.nb_degrees)],
                value=self.initial_selected_degree,
                labelStyle={'display': 'inline-block'}
            ),
        ]

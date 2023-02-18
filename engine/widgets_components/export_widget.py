from __future__ import annotations
from types import FunctionType
from typing import List
from dash import html, dcc, ctx
from dash.development.base_component import Component
from engine.models import Model
import engine.utils.utils as utils
import sys
from engine.widgets_abstract import Widget
import logging


class ExportView(Widget):
    def __init__(self) -> None:
        super().__init__("export-df-button",
                         {
                             "graph": {
                                 "input": ["n_clicks"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.Button('Export filtered data to CSVs', id=self.id)
        ]

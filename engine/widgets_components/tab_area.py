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


class TabsBar(Widget):
    def __init__(self) -> None:
        super().__init__("tabs-bar",
                         {
                             "graph": {
                                 "input": ["value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return dcc.Tabs(id=self.id, value=list(Tab.tabs.values())[0].name, children=[tab.get_html_components() for tab in Tab.tabs.values()])


class TabsContent(Widget):
    def __init__(self) -> None:
        super().__init__(
            "tabs-content",
            {
                "graph": {
                    "output": ["children"]
                }
            })

    def get_html_components(self) -> list[Component]:
        return [html.Div(id=self.id)]

    def _calculate_output(self, inputs) -> any:
        tabs_value = inputs["tabs-bar"]
        return Tab.tabs[tabs_value].output(inputs)

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


class SliceLabelRankingSelector(Widget):
    def __init__(self) -> None:
        super().__init__("slice-label-ranking-selector",
                         {
                             "graph": {
                                 "input": ["value"],
                             },
                             "state_update": {
                                 "input": ["value"],
                                 "output": ["options"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Slice label ranking'),
            dcc.Dropdown(id=self.id)
        ]

    def _calculate_output(self, inputs: dict[str: any]) -> any:
        model1 = self.selected_models[0]
        if model1 is None:
            return []
        return list(c[len(self.settings.SORT_ID):] for c in model1.slice_df.columns if self.settings.SORT_ID in c)

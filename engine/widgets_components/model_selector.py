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


class ModelSelector(Widget):
    def __init__(self) -> None:
        super().__init__("model-selector",
                         {
                             "state_update": {
                                 "input": ["value"] * self.settings.NB_MODELS,
                                 "output": ["disabled"] * self.settings.NB_MODELS
                             }
                         })
        self.available_models: list[Model] = [Model(x.name) for x in self.settings.MODEL_ROOT_FOLDER.glob(
            '?*') if x.is_dir()]
        self.nb_models = self.settings.NB_MODELS

    def get_html_components(self) -> list[Component]:
        return [html.Div([html.H3([f'Model {i+1}:']), dcc.Dropdown(
            id=f'{self.id}{i+1}', options=[{'label': m.name, 'value': m.name} for m in self.available_models], value='', className='model-selector-box'),
        ], className='model-selector-zone') for i in range(self.nb_models)]

    def _calculate_output(self, inputs) -> tuple[bool]:
        # read inputs
        model_selection = inputs["model-selector"]

        # we have to return a tuple containing wether or not the model-selector is disabled
        dropdown_disabled = [False] + [True] * (self.nb_models - 1)
        for index, model_name in enumerate(model_selection):
            if bool(model_name):
                # Only load data if model_name has changed
                if (self.selected_models[index] is None) or (self.selected_models[index].name != model_name):
                    self.selected_models[index] = self.settings.MODEL_CLASS(model_name, load=True)

                # disable previous selectors
                if index > 0:
                    dropdown_disabled[index - 1] = True
                # enable next selectors
                if index < self.nb_models - 1:
                    dropdown_disabled[index + 1] = False

            elif self.selected_models[index] is not None:
                self.selected_models[index] = None

                # enable previous selectors
                if index > 0:
                    dropdown_disabled[index - 1] = False
                # disable next selectors
                if index < self.nb_models - 1:
                    dropdown_disabled[index + 1] = True

        return tuple(dropdown_disabled)

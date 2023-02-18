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


class FilterTab(Tab):
    def __init__(self) -> None:
        super().__init__("Filter")

    def _calculate_output(self, inputs) -> any:
        return []

    def get_html_components(self):
        children = [
            html.Div([
                html.Div(Widget.widgets["graph-type-selector"].get_html_components(),
                         style={'width': '20%', 'display': 'inline-block'}),
                html.Div(Widget.widgets["axis-label-selector"].get_html_components(),
                         style={'width': '50%', 'display': 'inline-block'})
            ], style={"display": "flex"}),
            html.Div(Widget.widgets["size-threshold"].get_html_components()),
            html.Div(
                Widget.widgets["slice-label-value-filter"].get_html_components()),
            html.Div([
                html.Div(Widget.widgets["sort-label-property"].get_html_components(),
                         style={'width': '55%', 'display': 'inline-block'}),
                html.Div(Widget.widgets["sort-model-selector"].get_html_components(),
                         style={'width': '35%', 'display': 'inline-block'}),
                html.Div(Widget.widgets["sort-label-property-operator"].get_html_components(),
                         style={'width': '10%', 'display': 'inline-block'})
            ]),
        ]
        return super().get_html_components(children)


class SizeThreshold(Widget):
    def __init__(self) -> None:
        super().__init__("size-threshold",
                         {
                             "graph": {
                                 "input": ["value"],
                                 "output": ["max"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Size Threshold'),
            dcc.RangeSlider(0, sys.maxsize, value=[0, sys.maxsize],
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                            id=self.id)
        ]

    def _calculate_output(self, inputs) -> any:
        max = 0
        for model in [model for model in self.selected_models if model is not None]:
            if model.slice_df["size"].max() > max:
                max = model.slice_df["size"].max()
        return max


class SliceLabelValueFilter(Widget):
    def __init__(self) -> None:
        super().__init__("slice-label-value-filter",
                         {
                             "graph": {
                                 "input": ["value"],
                                 "output": ["options", "value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Filter label values'),
            dcc.Dropdown(multi=True, id=self.id)
        ]

    def _calculate_output(self, inputs) -> any:
        # we assume only labels can be selected that are in the intersection of all models
        input_values = inputs["slice-label-value-filter"]
        options = []

        slice_label_value = inputs["slice-label-selector"]
        if slice_label_value is not None:

            slice_labels = slice_label_value.split(' & ')

            if self.selected_models[0] is not None:
                for slice_label in slice_labels:
                    values = self.selected_models[0].slice_df[slice_label].dropna(
                    ).unique()
                    for value_ in values:
                        # remove trailing zeros
                        try:
                            value_ = int(value_)
                        except ValueError:
                            pass
                        options.append(f'{slice_label}:{value_}')

        # delete values that are not in options (anymore)
        if input_values is not None:
            for value in input_values:
                if value not in options:
                    input_values.remove(value)
        return {"options": options, "value": input_values}


class SortLabelProperty(Widget):
    def __init__(self) -> None:
        super().__init__("sort-label-property",
                         {
                             "graph": {
                                 "input": ["value"],
                                 "output": ["options", "value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Sorting (line graph type only)'),
            dcc.Dropdown(id=self.id, multi=True)
        ]

    def _calculate_output(self, inputs) -> any:
        graph_type = inputs['graph-type-selector']
        disabled = graph_type == 'array'

        input_values = inputs["sort-label-property"]
        options = list()

        slice_label_value = inputs["slice-label-selector"]
        if slice_label_value is not None:
            slice_labels = slice_label_value.split(' & ')
            for label in slice_labels:
                options.append({"label": label, "value": label, "disabled": disabled})

        options += [{"label": column, "value": column, "disabled": disabled}
                    for column in self.settings.EXTRA_SORTING_COLUMN]
        options += [{"label": column,
                     "value": self.settings.METRIC_OPTIONS[column]["column_name"],
                     "disabled": disabled} for column in list(self.settings.METRIC_OPTIONS.keys())]

        possible_input_values = []
        for dic in options:
            possible_input_values.append(dic["value"])
        if input_values is not None:
            for value in input_values:
                if value not in possible_input_values:
                    input_values.remove(value)

        return {"options": options, "value": input_values}


class SortLabelOperator(Widget):
    def __init__(self) -> None:
        super().__init__("sort-label-property-operator",
                         {
                             "graph": {
                                 "input": ["value"],
                                 "output": ["options", "value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('order'),
            dcc.Dropdown(id=self.id, options=[
                         "0 -> Z", "Z -> 0"], value="0 -> Z")
        ]

    def _calculate_output(self, inputs) -> any:
        graph_type = inputs['graph-type-selector']
        disabled = graph_type == 'array'
        value = inputs[self.id]

        if disabled:
            return {"options": [{'label': "0 -> Z", 'value': "0 -> Z", 'disabled': True}], "value": "0 -> Z"}
        else:
            return {"options": ["0 -> Z", "Z -> 0"], "value": value}


class SortModelSelector(Widget):
    def __init__(self) -> None:
        super().__init__("sort-model-selector",
                         {
                             "graph": {
                                 "input": ["value"],
                             },
                             "state_update": {
                                 "input": ["value"],
                                 "output": ["options", "value"]
                             }

                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Model to sort'),
            dcc.Dropdown(id=self.id)
        ]

    def _calculate_output(self, inputs) -> any:
        value = inputs[self.id]
        options = [{"label": model.name, "value": model.name} for model in self.selected_models if model is not None]

        if (value not in options or value is None) and len(options) > 0:
            value = options[0]

        return {"options": options, "value": value}


class AxisLabelSelector(Widget):
    def __init__(self) -> None:
        super().__init__("axis-label-selector",
                         {
                             "graph": {
                                 "input": ["value"],
                                 "output": ["options", "value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Axis order ( x & y (& y2) )'),
            dcc.Dropdown(id=self.id, multi=True)
        ]

    def _calculate_output(self, inputs) -> any:
        input_values = inputs[self.id]
        options = list()

        slice_label_value = inputs["slice-label-selector"]
        if slice_label_value is not None:
            slice_labels = slice_label_value.split(' & ')
            for label in slice_labels:
                options.append(label)

        if input_values is not None:
            for value in input_values:
                if value not in options:
                    input_values.remove(value)

        return {"options": options, "value": input_values}


class GraphTypeSelector(Widget):
    def __init__(self) -> None:
        super().__init__("graph-type-selector",
                         {
                             "graph": {
                                 "input": ["value"],
                             }
                         })

    def get_html_components(self) -> list[Component]:
        return [
            html.H2('Graph type'),
            dcc.RadioItems(id=self.id, options=[
                           "line", "array"], value="array", inline=True)
        ]

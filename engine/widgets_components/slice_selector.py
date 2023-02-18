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


class SliceSelector(Widget):
    def __init__(self) -> None:
        super().__init__("slice-label-selector",
                         {
                             "graph": {
                                 "input": ["value"],
                             },
                             "state_update": {
                                 "output": ["options", "value"],
                                 "input": ["value"]
                             }
                         })

    def get_html_components(self) -> list[Component]:
        logging.info('SliceSelector get_html called')
        return [
            html.H2('Slice label'),
            dcc.Dropdown(id=self.id)
        ]

    def _calculate_output(self, inputs) -> any:
        output = {"options": [], "value": None}
        degree = inputs["degree-selector"]
        ranking_method = inputs["slice-label-ranking-selector"]
        input_value = inputs[self.id]
        model1 = self.selected_models[0]
        first_item = ''

        itemlist = []

        if model1 is None:
            return output

        slice_df = model1.slice_df[model1.slice_df['degree'] == degree]

        # sort based on certain metric
        sort_column = f'{self.settings.SORT_ID}{ranking_method}'
        if sort_column not in slice_df.columns:
            sort_column = self.settings.DEFAULT_SORT_COLUMN

        slice_df = slice_df.sort_values(sort_column)

        # Process the data from the data_slicing results so that metadata labels are extracted
        for item in slice_df['description'].unique():
            item = item.replace(':', ' ').split()[::2]
            itemstring = ''
            item.sort()
            itemstring = ' & '.join(item)
            # Only add the new option to the list if it isn't in there yet
            if itemstring not in itemlist:
                itemlist.append(itemstring)
                if first_item == '':
                    first_item = itemstring

        if input_value is None:
            selected_item = first_item
        elif input_value not in itemlist:
            selected_item = first_item
        else:
            selected_item = input_value

        output["options"] = itemlist
        output["value"] = selected_item
        return output

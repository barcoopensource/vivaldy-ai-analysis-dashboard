from __future__ import annotations

from dash import html, dcc
from dash.development.base_component import Component
from dash.dependencies import Input, Output, DashDependency
from engine.models import Model
from engine.properties import EngineProperties


class Widget(EngineProperties):
    """This Class is the base of every interactive component on the dashboard.
    """
    widgets: dict[str:Widget] = {}

    def __init__(self, component_id: str, callback_properties: dict[str:dict[str:list[str] | None]]) -> None:
        """Initialize this widget

        :param component_id: id which will be used to identify this widget
        :type component_id: str
        :param callback_properties: dictionary with as key the callback functions and as value another \\
            dictionary with as key in- or output and as value a list containing the proper properties for this widget:
            {
                "state_update": {
                    "input": None,
                    "output": None
                },
                "graph": {
                    "input": None,
                    "output": None
                }
            }
            NOTE:
            If a list contains multiple copies of the same property, then this widget will have multiple ids in the
            callbacks by indexing the given component_id. For example:
            callback_properties = {
                "state_update": {
                    "input": ["value", "value", "value"],
                    "output": None
                },
                "graph": {
                    "input": None,
                    "output": None
                }
            }
            results in the following callbacks (given to engine):
            {
                "state_update": {
                    "input": (Input(f"{self.id}1","value"), Input(f"{self.id}2","value"), Input(f"{self.id}3","value"))
                    "output": None
                },
                "graph": {
                    "input": None,
                    "output": None
                }
            }

            If a list contains different properties, the id doesn't change.

            Keep this in mind when overriding get_html_components.

        :type callback_properties: dict[str:dict[str:list[str]|None]]
        """
        self.id = component_id
        self.callback_properties = callback_properties
        self.widgets[self.id] = self

    def get_html_components(self) -> list[Component]:
        """Override this method to define the layout of this widget.
        Keep in mind that the ids used in the callbacks (self.id or self.id<index>)
        must be included in the html. """

    def get_callbacks(self) -> dict[str:dict[str:list[DashDependency]]]:
        """Get the dash dependencies to add this widget to the in- and outputs of callback functions .

        :return: This function returns a dict containing the in- and outputs for each callback function
        based on the callback_properties variable.

        If a list contains multiple copies of the same property, then this widget will have multiple ids in the
        callbacks by indexing the given component_id.
        For example:
        callback_properties = {
            "state_update": {
                "input": ["value", "value", "value"],
                "output": None
            },
            "graph": {
                "input": None,
                "output": None
            }
        }
        results in the following callbacks (given to engine):
        {
            "state_update": {
                "input": (Input(f"{self.id}1","value"), Input(f"{self.id}2","value"), Input(f"{self.id}3","value"))
                "output": None
            },
            "graph": {
                "input": None,
                "output": None
            }
        }

        If a list contains different properties, the id doesn't change.
        For example:
        callback_properties = {
            "state_update": {
                "input": ["value", "options"],
                "output": None
            },
            "graph": {
                "input": None,
                "output": None
            }
        }
        results in the following callbacks (given to engine):
        {
            "state_update": {
                "input": None
                "output": (Output(self.id,"value"), Output(self.id, "options"))
            },
            "graph": {
                "input": None,
                "output": None
            }
        }
        :rtype: dict[str:dict[str:list[DashDependency]]]
        """
        output = {
            "state_update": {
                "input": None,
                "output": None
            },
            "graph": {
                "input": None,
                "output": None
            }
        }
        for callback in output:  # pylint: disable=consider-using-dict-items
            if callback in self.callback_properties:
                if "input" in self.callback_properties[callback]:
                    properties = self.callback_properties[callback]["input"]
                    # if multiple properties are given, check if they are uniqe or not, give an extra index to their id if not uniqe
                    if len(set(properties)) > 1:
                        properties_count = [(x, properties.count(x)) for x in set(properties)]
                        output[callback]["input"] = {}
                        for property, count in properties_count:
                            # properties that count multiple times will have an extra index to their id
                            output[callback]["input"][property] = tuple([Input(f'{self.id}{i+1}', property) for i in range(count)]) \
                                if count > 1 else Input(f'{self.id}', property)
                    else:
                        output[callback]["input"] = tuple([Input(f'{self.id}{i+1}', properties[0]) for i in range(len(properties))]) \
                            if len(properties) > 1 else Input(f'{self.id}', properties[0])

                if "output" in self.callback_properties[callback]:
                    properties = self.callback_properties[callback]["output"]
                    # if multiple properties are given, check if they are uniqe or not, give an extra index to their id if not uniqe
                    if len(set(properties)) > 1:
                        properties_count = [(x, properties.count(x)) for x in set(properties)]
                        output[callback]["output"] = {}
                        for property, count in properties_count:
                            output[callback]["output"][property] = tuple([Output(f'{self.id}{i+1}', property) for i in range(count)]) \
                                if count > 1 else Output(f'{self.id}', property)
                    else:
                        output[callback]["output"] = tuple([Output(f'{self.id}{i+1}', properties[0]) for i in range(len(properties))]) \
                            if len(properties) > 1 else Output(f'{self.id}', properties[0])

        return output

    def output(self, inputs: dict[str:any] = {}) -> any:
        """This function returns the output of this widget

        :param inputs: inputs is a dict containing all the inputs the callback which this widgets outputs to receives, defaults to {}
        :type inputs: dict[str:any], optional
        :return: the output of this widget
        :rtype: any
        """

        return self._calculate_output(inputs)

    def _calculate_output(self, inputs) -> any:
        """Override this method if the widget has an output."""


class Tab(EngineProperties):
    """This class is the base of every Tab."""

    # dict containting all the tabs made
    tabs: dict[str:Tab] = {}

    def __init__(self, name: str) -> None:
        """Initialize this tab

        :param name: name of the tab, as will be shown in the tabs bar
        :type name: str
        """
        self.name = name
        self.tabs[name] = self

    def get_html_components(self, children=None):
        """Returns the html component of this tab."""
        return dcc.Tab(label=self.name, value=self.name, children=children)

    def output(self, inputs: dict[str:any] = {}) -> any:
        """This function returns the output of this widget

        :param inputs: inputs is a dict containing all the inputs the callback which this tab outputs to, receives, defaults to {}
        :type inputs: dict[str:any], optional
        :return: the output of this widget
        :rtype: any
        """

        return self._calculate_output(inputs)

    def _calculate_output(self, inputs) -> any:
        """Override this method to give this tab an output."""

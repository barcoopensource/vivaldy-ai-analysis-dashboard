from dash import Dash
from dash.dependencies import DashDependency
from engine.layout import Layout
from engine.widgets import *
import logging
from engine.properties import EngineProperties


class Engine(EngineProperties):
    """This class initializes all the tabs & widgets defined in settings, \\
    collects all callback inputs and outputs and generates the output for the \\
    callback functions defined in Dashboard
    """

    def __init__(self) -> None:
        # initialize tabs & widgets
        self.settings.initialize_tabs()
        self.settings.initialize_widgets()

        # define callback inputs and outputs
        self.update_state_cb_inputs: dict[str:DashDependency | tuple[DashDependency]] = {}
        self.update_state_cb_outputs: dict[str:DashDependency | tuple[DashDependency]] = {}
        self.graph_cb_inputs: dict[str:DashDependency | tuple[DashDependency]] = {}
        self.graph_cb_outputs: dict[str:DashDependency | tuple[DashDependency]] = {}

        # load in callbacks
        for widget in Widget.widgets.values():
            if widget.get_callbacks()["state_update"]["input"] is not None:
                self.update_state_cb_inputs[widget.id] = widget.get_callbacks()["state_update"]["input"]
            if widget.get_callbacks()["state_update"]["output"] is not None:
                self.update_state_cb_outputs[widget.id] = widget.get_callbacks()["state_update"]["output"]
            if widget.get_callbacks()["graph"]["input"] is not None:
                self.graph_cb_inputs[widget.id] = widget.get_callbacks()["graph"]["input"]
            if widget.get_callbacks()["graph"]["output"] is not None:
                self.graph_cb_outputs[widget.id] = widget.get_callbacks()["graph"]["output"]

        logging.debug("Update State Input Callbacbks %s", self.update_state_cb_inputs)
        logging.debug("Update State Output Callbacbks %s", self.update_state_cb_outputs)
        logging.debug("Graph Input Callbacbks %s", self.graph_cb_inputs)
        logging.debug("Graph Output Callbacbks %s", self.graph_cb_outputs)

        # define the dash app
        self.app: Dash = Dash(__name__, suppress_callback_exceptions=True, assets_folder=self.settings.ASSETS_FOLDER)
        self.app.title = self.settings.TITLE
        self.layout = Layout(self)
        self.app.layout = self.layout.index()

    def update_state(self, inputs):
        logging.debug("Inputs update_state: %s", inputs)
        output = {}
        for key in self.update_state_cb_outputs:
            # call output function and save to output
            output[key] = Widget.widgets[key].output(inputs)
        return output

    def graph_callback(self, inputs):
        logging.debug("Inputs graph_callback: %s", inputs)
        output = {}
        for key in self.graph_cb_outputs:
            # call output function and save to output
            output[key] = Widget.widgets[key].output(inputs)
        return output

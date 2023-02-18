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

# top UI Widgets
from engine.widgets_components.degree_selector import DegreeSelector
from engine.widgets_components.slice_selector import SliceSelector
from engine.widgets_components.slice_label_ranking_selector import SliceLabelRankingSelector

# Bottom UI Widgets
from engine.widgets_components.model_selector import ModelSelector

# Left UI Widgets
from engine.widgets_components.legend import Legend
from engine.widgets_components.metric_selector import MetricSelector

# Filter tab widgets
from engine.widgets_components.filter_tab import FilterTab, \
    SizeThreshold, SliceLabelValueFilter, SortLabelProperty, SortLabelOperator, SortModelSelector, \
    AxisLabelSelector, GraphTypeSelector

# Tab area
from engine.widgets_components.tab_area import TabsBar, TabsContent

# Plot Tabs
from engine.widgets_components.performance_tab import PerformanceTab
from engine.widgets_components.samples_tab import SamplesTab
from engine.widgets_components.multithreshold_curve_tab import MultithresholdCurveTab

# Export Widget
from engine.widgets_components.export_widget import ExportView

from __future__ import annotations
import logging
from typing import List


from engine.plots import *
from engine.utils.file_io import *


def list_intersection(lists) -> list:
    all_elements = list(set.intersection(*[set(e) for e in lists]))
    return all_elements


def list_union(lists) -> list:
    all_elements = list(set.union(*[set(e) for e in lists]))
    return all_elements


def parse_axis_order(axis_order) -> List:
    if axis_order is None:
        return []
    else:
        return axis_order


def extract_label_options(inputs: dict) -> dict:
    label_value_filter = inputs["slice-label-value-filter"]
    label_options_candidates = {}
    if label_value_filter is None:
        return label_options_candidates

    for label_value in inputs["slice-label-value-filter"]:
        label, value = label_value.split(':')
        # change str to float if value is was a float/int
        try:
            value = float(value)
        except Exception as e:
            pass
        if label in label_options_candidates:
            label_options_candidates[label].append(value)
        else:
            label_options_candidates[label] = [value]

    # Only keep label options present in currently selected slice label:
    slice_labels = inputs["slice-label-selector"].split(' & ')
    label_options = {}
    for k, v in label_options_candidates.items():
        if k in slice_labels:
            label_options[k] = v
    return label_options


def filter_and_sort_models(selected_models: List[Model], inputs: dict) -> List[Model]:
    import operator

    degree = inputs["degree-selector"]
    slice_label = inputs["slice-label-selector"]
    size_threshold = inputs["size-threshold"]
    sort_operator = inputs["sort-label-property-operator"]
    sort_columns = inputs["sort-label-property"]
    sort_model_selector = inputs["sort-model-selector"]
    axis_order = parse_axis_order(inputs["axis-label-selector"])
    ranking_method = inputs["slice-label-ranking-selector"]

    models = [model for model in selected_models if model is not None]
    label_options = extract_label_options(inputs)
    # set model to sort on as the first model
    if sort_model_selector in [model.name for model in selected_models if model is not None]:
        sort_model_index = [model.name for model in selected_models if model is not None].index(
            sort_model_selector)
        models[0], models[sort_model_index] = models[sort_model_index], models[0]

    loaded_models = []
    for model in models:
        model.filter_degree_slices(degree, slice_label, ranking_method)
        model.filter_threshold("size", size_threshold[0], operator.ge)
        model.filter_threshold("size", size_threshold[1], operator.le)
        for label in label_options:
            model.filter_contains_values(label, label_options[label])
        if sort_columns is not None and sort_columns != []:
            model.sort(sort_columns, sort_operator == "0 -> Z")
        else:
            if len(axis_order) != degree:
                axis_order = slice_label.split(' & ')
            model.sort(axis_order, sort_operator == "0 -> Z")

        if not model.slice_df_filtered.empty:
            loaded_models.append(model)
    return loaded_models

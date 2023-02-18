from __future__ import annotations
from audioop import mul
from statistics import multimode
from types import FunctionType
from typing import List
from dash import html, dcc, ctx
from dash.development.base_component import Component
from engine.models import Model
import engine.utils.utils as utils
import sys
from engine.widgets_abstract import Widget, Tab
import logging


class MultithresholdCurveTab(Tab):
    def __init__(self) -> None:
        super().__init__("ROC Performance per slice")

    def _calculate_output(self, inputs) -> any:
        import plotly.graph_objects as go
        import sklearn
        import itertools
        import cycler
        import matplotlib.pyplot as plt
        import json
        import numpy as np
        from PIL import ImageColor
        color_cycler: cycler.Cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'])
        # get inptus
        metric = inputs["metric-selector"]
        degree = inputs["degree-selector"]
        slice_label = inputs["slice-label-selector"]
        size_threshold = inputs["size-threshold"]
        label_value_filter = inputs["slice-label-value-filter"]
        sort_operator = inputs["sort-label-property-operator"]
        sort_columns = inputs["sort-label-property"]
        axis_order = utils.parse_axis_order(inputs["axis-label-selector"])
        graph_type = inputs["graph-type-selector"]
        sort_model_selector = inputs["sort-model-selector"]

        label_options = {}
        if label_value_filter is not None:
            for label_value in inputs["slice-label-value-filter"]:
                label, value = label_value.split(':')
                # change str to float if value is was a float/int
                try:
                    value = float(value)
                except Exception as e:
                    pass
                if label in label_options:
                    label_options[label].append(value)
                else:
                    label_options[label] = [value]

        loaded_models = utils.filter_and_sort_models(self.selected_models, inputs)

        df = loaded_models[0].slice_df_filtered.copy()
        fig = go.Figure()

        slice_labels = slice_label.split(' & ')
        slice_label_options = {}
        for label in slice_labels:
            slice_label_options[label] = df[label].unique()

        for sel_tuple, color_options in zip(itertools.product(*slice_label_options.values()), color_cycler):
            line_options = color_options.copy()
            df_selected = df.copy(deep=True)
            name = ''
            for column_name, slice_value in zip(slice_labels, sel_tuple):
                df_selected = df_selected[df_selected[column_name] == slice_value]
                name += f'{column_name}:{slice_value}  '

            if 'ROC' not in df_selected.columns:
                multithreshold_data = self._calculate_ROC(loaded_models, label_options, slice_labels, sel_tuple)
            else:
                multithreshold_data = json.loads(df_selected['ROC'].iloc[0])
                multithreshold_data['fpr_mean'] = multithreshold_data['fpr']

            fpr = multithreshold_data['fpr']
            tpr = multithreshold_data['tpr']

            # plotting section
            try:
                fig.add_scatter(x=fpr, y=tpr, mode='lines', name=name, line=line_options)
            except ValueError as e:
                pass

            # add CI
            if self.settings.ROC_PERFORMANCE['show_CI'] and all(c in multithreshold_data.keys() for c in ['TPR_LB', 'TPR_UB']):
                tpr_LB = multithreshold_data['TPR_LB']
                tpr_UB = multithreshold_data['TPR_UB']
                fpr_mean = multithreshold_data['fpr_mean']

                x_err = np.hstack([fpr_mean, fpr_mean[::-1]])
                y_err = np.hstack([tpr_UB, tpr_LB[::-1]])

                fillcolor = list(ImageColor.getcolor(line_options['color'], 'RGB'))
                fillcolor.append(0.3)
                fillcolor = 'rgba({},{},{},{})'.format(*fillcolor)
                fig.add_scatter(x=x_err, y=y_err, mode='none', line=line_options, name=name, fill='toself', fillcolor=fillcolor)

        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='reference', line=go.scatter.Line(dash='longdash', color='black'))
        fig.layout = go.Layout(xaxis=dict(title='1 - Specificity', showgrid=True, showline=True),
                               yaxis=dict(title='Sensitivity', showgrid=True, showline=True))
        if fig is None:
            return []

        if "export-df-button" == ctx.triggered_id:
            utils.export_to_csvs(loaded_models, inputs, 'MultithresholdCurveTab')

        return [html.Div([
            html.H2('ROC'),
            dcc.Graph(id='multithreshold_curve', figure=fig)
        ])]

    def _calculate_ROC(self, loaded_models, label_options, slice_labels, sel_tuple) -> dict:
        import numpy as np
        import sklearn
        df_full = loaded_models[0].full_df.copy(deep=True)
        for label in label_options:
            df_full = df_full[df_full[label].isin(label_options[label])]
        for column_name, slice_value in zip(slice_labels, sel_tuple):
            df_full = df_full[df_full[column_name] == slice_value]
        gt = df_full['malignancy_label_gt']
        pred = df_full['malignancy_label_pred']
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=gt, y_score=pred)
        multithreshold_data = {}
        multithreshold_data['fpr'] = fpr
        multithreshold_data['tpr'] = tpr
        multithreshold_data['thresholds'] = thresholds
        if self.settings.ROC_PERFORMANCE['show_CI'] and self.settings.ROC_PERFORMANCE['allow_CI_bootstrapping_fallback']:
            n_resamples = self.settings.ROC_PERFORMANCE['CI_resamples']
            gt_CI = np.array(gt)
            pred_CI = np.array(pred)
            original_idx = list(range(len(pred_CI)))
            resampled_data = []
            for n in range(n_resamples):
                idx_random = np.random.choice(original_idx, len(original_idx), replace=True)
                gt_random = gt_CI[idx_random]
                pred_random = pred_CI[idx_random]
                resampled_data.append([gt_random, pred_random])

            fpr_list = []
            tpr_list = []
            thresholds_list = []
            for x, y in resampled_data:
                fpr_t, tpr_t, thresholds_t = sklearn.metrics.roc_curve(x, y)
                fpr_list.append(fpr_t)
                tpr_list.append(tpr_t)
                thresholds_list.append(thresholds_t)

            fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt, pred)
            threhold_r = thresholds
            threshold_max = threhold_r[:-1]
            threshold_min = threhold_r[1:]

            tpr_CI_UB = []
            tpr_CI_LB = []
            fpr_mean = []

            tpr_CI_UB.append(0)
            tpr_CI_LB.append(0)
            fpr_mean.append(0)
            for t_min, t_max in zip(threshold_min, threshold_max):
                fpr_ci_list = []
                tpr_ci_list = []
                for fpr_t, tpr_t, threshold_t in zip(fpr_list, tpr_list, thresholds_list):
                    sel = (threshold_t >= t_min) & (threshold_t <= t_max)
                    for x in fpr_t[sel]:
                        fpr_ci_list.append(x)
                    for x in tpr_t[sel]:
                        tpr_ci_list.append(x)
                if any(sel):
                    tpr_CI_UB.append(np.percentile(tpr_ci_list, 97.5))
                    tpr_CI_LB.append(np.percentile(tpr_ci_list, 2.5))
                    fpr_mean.append(np.mean(fpr_ci_list))

            tpr_CI_UB.append(1.0)
            tpr_CI_LB.append(1.0)
            fpr_mean.append(1.0)

            multithreshold_data['TPR_LB'] = tpr_CI_LB
            multithreshold_data['TPR_UB'] = tpr_CI_UB
            multithreshold_data['fpr_mean'] = fpr_mean
        return multithreshold_data

from __future__ import annotations
import json
import pandas
from typing import TYPE_CHECKING, List
from engine.utils import utils
if TYPE_CHECKING:
    from engine.models import Model


def export_to_csvs(models: list[Model], inputs: dict, origin: str) -> None:
    """Save filtered dataframe to csv"""
    import dashboard_settings as dashboard_settings
    import time
    import itertools

    slice_label = inputs["slice-label-selector"]

    export_path = dashboard_settings.EXPORTS_FOLDER / \
        f'export{time.strftime("%Y%m%d_%H%M")}'

    for m in models:
        df = m.slice_df_filtered
        full_df = m.full_df
        model_name = m.name

        path = export_path / model_name
        path.mkdir(parents=True, exist_ok=True)

        labels = slice_label.split(' & ')
        label_values = []
        for x in labels:
            label_value_pairs = []
            for y in df[x].unique():
                label_value_pairs.append({x: y})
            label_values.append(label_value_pairs)
        for conditions in itertools.product(*label_values):
            df_subset = full_df.copy()
            condition_list = []
            for c in conditions:
                for k, v in c.items():
                    df_subset = df_subset[df_subset[k] == v]
                    condition_list.append(str(k) + '==' + str(v))
            file_name = ' & '.join(condition_list)
            if not df_subset.empty:
                df_subset.to_csv(path / f"full_metadata_{file_name}.csv")

        # export rows with all other relevant info (such as metric values and sample sizes) for the selected slice_label
        df_subset = df.copy()
        for x in labels:
            df_subset = df_subset[~df_subset[x].isna()]
        df_subset.to_csv(path / f"metrics - {' & '.join(labels)}.csv")

    if origin == 'MultithresholdCurveTab':
        """
        Additionally export:
                    Individual ROC x,y datapoints + CI
        """
        pass

    # Export filter settings into a json file
    inputs['models'] = list(m.name for m in models)
    with open(str(export_path / 'settings.json'), 'w') as f:
        json.dump(inputs, f)

    return None

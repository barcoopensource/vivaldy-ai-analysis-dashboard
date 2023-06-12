from pathlib import Path
import pandas as pd
from types import ModuleType, FunctionType
from typing import List
from data_slicing_pipeline.data_slicing_pipeline_abstract import DataSlicingPipelineAbstract
import os


def run_data_slicing_experiment(config: ModuleType,
                                output_name,
                                df: pd.DataFrame,
                                degrees=[0, 1, 2, 3],
                                optional_preprocessing_steps: List[FunctionType] = [],
                                stop_after_df_preparation=False):
    data_output_dir = Path(os.path.abspath('')) / 'models' / f'{output_name}'

    for preprocessing_function in optional_preprocessing_steps:
        df = preprocessing_function(df)

    DATA_IDX = config.settings.get('dataframe').get('index')
    df = df.set_index(DATA_IDX)

    Pipeline: DataSlicingPipelineAbstract = config.settings.get('pipeline')

    sfp = Pipeline(data_output_dir, degrees, config)
    sfp.process(df, stop_after_df_preparation)

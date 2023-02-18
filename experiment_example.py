from pathlib import Path
import pandas as pd
from types import ModuleType, FunctionType
from typing import List
from data_slicing_pipeline.data_slicing_pipeline_abstract import DataSlicingPipelineAbstract
from pathlib import Path


def run_data_slicing_experiment(config: ModuleType, output_name, df: pd.DataFrame, degrees=[0, 1, 2, 3], optional_preprocessing_steps: List[FunctionType] = []):
    data_output_dir = Path(__file__).parent / 'models' / f'{output_name}'

    for preprocessing_function in optional_preprocessing_steps:
        df = preprocessing_function(df)

    DATA_IDX = config.settings.get('dataframe').get('index')
    df = df.set_index(DATA_IDX)

    Pipeline: DataSlicingPipelineAbstract = config.settings.get('pipeline')

    sfp = Pipeline(data_output_dir, degrees, config)
    sfp.process(df)


def _data_parsing(df: pd.DataFrame) -> pd.DataFrame:
    # fix incorrectly saved file if necessary
    if 'level_0' in df.columns:
        df = df.drop(columns='level_0')

    # remove records without gt
    df = df[~pd.isna(df['malignant'])]
    return df


if __name__ == '__main__':
    import settings.settings_data_slicing_pipeline_example as config

    experiments_to_run = [
        {
            'config': config,
            'df': pd.read_csv('input_file.csv', low_memory=False),
            'output_name': 'sliced_model',
            'optional_preprocessing_steps': [_data_parsing],
            'degrees': [0, 1, 2, 3]
        }
    ]

    for experiment in experiments_to_run:
        run_data_slicing_experiment(**experiment)

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_slicing_pipeline.data_slicing_pipeline_abstract import DataSlicingPipelineAbstract
import pandas as pd


class PostProcessingAbstract():
    @classmethod
    def process(cls, sfp: DataSlicingPipelineAbstract, postprocessing_settings: dict) -> pd.DataFrame:
        """To be implemented in child class"""
        pass

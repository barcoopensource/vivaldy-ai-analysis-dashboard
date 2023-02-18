"""Inspired by https://github.com/yeounoh/slicefinder"""
from typing import List
from data_slicing.data_slice import DataSlice
import pandas as pd


class DataSlicing:
    minimum_samples: int

    @classmethod
    def __init__(cls, y_predict, data: pd.DataFrame, minimum_samples: int, mp_settings):
        import itertools
        if mp_settings.get('use_mp'):
            import multiprocessing
            import functools
            cls.pool = multiprocessing.Pool(mp_settings.get('N_threads'))
            cls.map = functools.partial(cls.pool.imap, chunksize=mp_settings.get('chunksize'))
            cls.starmap = functools.partial(cls.pool.starmap, chunksize=mp_settings.get('chunksize'))
        else:
            cls.pool = None
            cls.map = map
            cls.starmap = itertools.starmap
        cls.y_predict = y_predict
        cls.data: pd.DataFrame = data
        cls.minimum_samples: int = minimum_samples

    def find_slice(self, degree: int) -> List[DataSlice]:
        ''' Find interesting subpopulations '''
        slices = self.slicing(degree)
        return slices

    @classmethod
    def slicing(cls, degree: int) -> List[DataSlice]:
        import itertools
        minimum_samples = cls.minimum_samples
        X = cls.data[0]

        # per column, get unique values. Create slice per column, per unique value
        col_slices = []
        for col in X.columns:
            slices = []
            for v in X[col].unique():
                s = DataSlice.from_dataframe(X, col, v)
                slices.append(s)
            col_slices.append(slices)

        # get combinations over columns, paired by degree
        column_combinations = list(itertools.combinations(range(len(col_slices)), degree))
        print(f'Number of column combinations found: {len(column_combinations)}')

        # for each combination, take the product over the iterables
        slice_combinations = []
        for column_combination in column_combinations:
            slice_combinations += list(itertools.product(*(col_slices[x] for x in column_combination)))
        print(f'Total number of sharding combinations found: {len(slice_combinations)}')

        # intersect within the combinations
        crossed_slices = list(cls.map(cls._create_slice_combo, slice_combinations))

        # filter out slices with less samples than the set minimum
        fill = cls.starmap(cls._check_minimum_size, zip(crossed_slices, [cls.minimum_samples] * len(crossed_slices)))

        # NOTE: here, additional filters can be added
        crossed_slices = list(itertools.compress(crossed_slices, fill))

        print(f'Total number of slices with #elements >= {minimum_samples}: {len(crossed_slices)}')
        return crossed_slices

    @staticmethod
    def _create_slice_combo(slices: List[DataSlice]) -> DataSlice:
        slice_ij = slices[0].deepcopy(clean=True)
        for slice in slices[1:]:
            slice_ij = slice_ij.intersection(slice)
        return slice_ij

    @staticmethod
    def _check_minimum_size(s: DataSlice, minimum_samples) -> bool:
        return s.size >= minimum_samples

    def __del__(self):
        if self.pool is not None:
            self.pool.close()

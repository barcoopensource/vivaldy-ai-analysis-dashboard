"""Inspired by https://github.com/yeounoh/slicefinder"""
from __future__ import annotations
from pandas import Index, DataFrame


def _new_DataSlice(d: dict):
    from pandas.core.indexes.base import _new_Index
    attrs = d.pop('attrs')
    idx: Index = _new_Index(Index, d)
    ds = DataSlice(idx)
    ds.attrs = attrs
    return ds


class DataSlice(Index):

    def __new__(cls, data=None, dtype=None, copy=False, name=None, tupleize_cols=True, **kwargs) -> DataSlice:
        slice_index = super().__new__(cls, data, dtype, copy, name, tupleize_cols, **kwargs)
        ds: DataSlice = object.__new__(cls)
        ds.__dict__ = slice_index.__dict__.copy()
        return ds

    def __reduce__(self):
        d = {"data": self._data, "name": self.name, 'attrs': self.attrs}
        return _new_DataSlice, (d,), None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._init_attributes()

    @classmethod
    def from_dataframe(cls, X: DataFrame, col: str = None, v=None) -> DataSlice:
        assert ((col is None) == (v is None))
        if (col is not None) and (v is not None):
            idx = X[X[col] == v].index
        else:
            idx = X.index
        s = DataSlice(idx)
        if (col is not None) and (v is not None):
            s.attrs['slice_info'][col] = v
        return s

    def _init_attributes(self):
        self.attrs = {'slice_info': {}, 'metric_values': {}}

    def _copy_attributes(self, other: DataSlice) -> None:
        from copy import deepcopy as dc
        self.attrs = dc(other.attrs)

    def _reset_metric_values(self) -> None:
        self.attrs['metric_values'] = {}

    def intersection(self, other: DataSlice, sort: bool = False) -> DataSlice:
        idx: Index = super().intersection(other, sort)
        s = DataSlice(idx)
        s._init_attributes()
        s._copy_attributes(self)

        for k, v in other.attrs['slice_info'].items():
            s.attrs['slice_info'][k] = v
        s._reset_metric_values()
        return s

    def deepcopy(self, clean=False) -> DataSlice:
        from copy import deepcopy as dc
        s = dc(self)
        s.__class__ = type(self)
        s._copy_attributes(self)
        if clean:
            s._reset_metric_values()
        return s

    def update_metric_value(self, metric_name, value) -> None:
        self.attrs['metric_values'][metric_name] = value

    def get_metric_value(self, metric_name) -> float:
        return self.attrs['metric_values'][metric_name]

    def get_all_metrics(self) -> dict:
        return self.attrs['metric_values']

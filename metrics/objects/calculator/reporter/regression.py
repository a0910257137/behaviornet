import dataclasses
from collections import defaultdict
from numbers import Number
from typing import Union

import numpy as np

from .base import Reporter

ROUND = 3


@dataclasses.dataclass
class RegressorReporter(Reporter):
    _count: dict = dataclasses.field(default_factory=lambda: defaultdict(int))
    _mean: dict = dataclasses.field(default_factory=lambda: defaultdict(float))
    _variance: dict = dataclasses.field(
        default_factory=lambda: defaultdict(float))
    __default_key__: str = 'core'

    def reset(self):
        self._count = 0
        self._mean = 0
        self._variance = 0.

    @property
    def report(self):
        return {
            'mean': {
                k: round(v, ROUND) if not np.isnan(v) else v
                for k, v in self.mean.items()
            },
            'std': {
                k: round(v, ROUND) if not np.isnan(v) else v
                for k, v in self.std.items()
            }
        }

    @property
    def count(self):
        return self._count

    @property
    def mean(self):
        return {
            k: self._mean[k] if self._count[k] != 0 else np.nan
            for k in self._count.keys()
        }

    @property
    def std(self):
        return {
            k: np.sqrt(self._variance[k]) if self._count[k] > 1 else np.nan
            for k in self._count.keys()
        }

    @property
    def var(self):
        return {
            k: self._variance[k] if self._count[k] > 1 else np.nan
            for k in self._count.keys()
        }

    @staticmethod
    def _update_mean(prev_count: Number, prev_mean: Number, curr_count: Number,
                     curr_mean: Number) -> Union[Number, Number]:
        """update sample mean value

        Args:
            prev_count (Number): previous number count
            prev_mean (Number): previous sample mean
            curr_count (Number): current input number count
            curr_mean (Number): current input number mean

        Returns:
            Union[Number, Number]: updated sample number count, updated sample mean
        """
        assert isinstance(prev_count, Number)
        assert isinstance(
            curr_count, Number
        ) and curr_count > 0, f'current count should be larger than 0'
        if prev_count == 0:
            # first time init
            return curr_count, curr_mean
        assert isinstance(prev_mean, Number) and prev_mean >= 0
        assert isinstance(curr_mean, Number) and curr_mean >= 0
        return prev_count + curr_count, (prev_count * prev_mean + curr_count *
                                         curr_mean) / (prev_count + curr_count)

    @staticmethod
    def _update_variance(prev_count: Number, prev_mean: Number,
                         prev_variance: Number, curr_count: Number,
                         curr_mean: Number, curr_variance: Number) -> Number:
        """update sample variance. Reference:
        https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-or-more-groups-given-known-group-varianc

        Args:
            prev_count (Number): previous number count
            prev_mean (Number): previous sample mean
            prev_variance (Number): previous sample variance
            curr_count (Number): current input number count
            curr_mean (Number): current input number mean
            curr_variance (Number): current input sample variance

        Returns:
            Number: updated sample variance
        """
        assert isinstance(prev_count, Number)
        assert isinstance(
            curr_count, Number
        ) and curr_count > 0, f'current count should be larger than 0'
        if prev_count == 0:
            # first time init
            return curr_variance

        assert isinstance(prev_mean, Number) and prev_mean >= 0
        assert isinstance(curr_mean, Number) and curr_mean >= 0
        assert isinstance(prev_variance, Number) and prev_variance >= 0
        assert isinstance(curr_variance, Number) and curr_variance >= 0
        return ((prev_count - 1) * prev_variance +
                (curr_count - 1) * curr_variance + prev_count * curr_count *
                ((prev_mean - curr_mean)**2) /
                (prev_count + curr_count)) / (curr_count + prev_count - 1)

    def update(self, value: Union[int, float, dict], **kwargs):
        assert isinstance(
            value, (dict, float,
                    dict)), f'Expected int, float, dict, Got {type(value)}'
        if isinstance(value, Number):
            value = {self.__default_key__: value}

        assert all(v >= 0 for v in value.values()
                   ), f'Expected to get all regression error are postives'

        for k, v in value.items():
            self._variance[k] = self._update_variance(self._count[k],
                                                      self._mean[k],
                                                      self._variance[k], 1, v,
                                                      0)
            self._count[k], self._mean[k] = self._update_mean(
                self._count[k], self._mean[k], 1, v)

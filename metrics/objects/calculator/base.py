import dataclasses
from abc import abstractmethod
from typing import ClassVar, List, Optional, Union

from bddhelper.objects import BaseLabelObject

from ..metric import Metric, MetricFactory
from ..transform import Compose, Transformer, TransformerFactory
from .reporter import Reporter, ReporterFactory


@dataclasses.dataclass
class ObjectCalculator:
    metric: Optional[Union[Metric, dict]] = None
    transform: Optional[Union[dict, Compose, Transformer, List[
        Union[Transformer, dict]]]] = None
    discarded_keys: Optional[Union[tuple, list]] = None
    reporter: Optional[Union[Reporter, dict]] = None
    __default_frame_id__: ClassVar[str] = '__all_frame__'

    def __post_init__(self):
        if self.metric is not None:
            assert isinstance(self.metric, (dict, Metric))
        self.metric = MetricFactory.create(
            **self.metric) if isinstance(self.metric, dict) else self.metric
        if not isinstance(self.transform, list) and self.transform is not None:
            self.transform = [self.transform]

        if self.transform is not None:
            _transform = []
            for trans in self.transform:
                if isinstance(trans, (Transformer, Compose)):
                    _transform.append(trans)
                else:
                    _transform.append(TransformerFactory.create(**trans))
            assert self.is_valid_transform(
                _transform
            ), f'Expected the last transform is a transform whose `ENT` is True'
            self.transform = Compose(_transform)

        if self.discarded_keys is not None:
            assert isinstance(self.discarded_keys, (list, tuple))

        if isinstance(self.reporter, dict):
            self.reporter = ReporterFactory.create(**self.reporter)
        else:
            assert isinstance(self.reporter, Reporter)

    @abstractmethod
    def __call__(self):
        # calculate what you want with matched, unmatched, raw objects
        pass

    @property
    @abstractmethod
    def report(self):
        # report of calculation
        pass

    @staticmethod
    def parse(lb: Union[Union[dict, BaseLabelObject], dict]):
        # gt_lb: Union[BDDObject, dict] or {'frame_id': Any, 'object': Union[BDDObject, dict]}
        assert isinstance(
            lb, (dict, BaseLabelObject)
        ), 'Expected lb in "{"frame_id":Any, "object": Union[dict, BaseLabelObject]}" or Union[dict, BaseLabelObject] directly, got lb: {lb}'
        try:
            return lb['object']
        except KeyError:
            return lb

    @staticmethod
    def is_valid_transform(transform: Union[Transformer, Compose]) -> bool:
        """check the transform is valid or not. In general case of ObjectCalculator, `GetShape` or `GetAttributes` are expected.
        """
        last_transform = transform[-1]
        if isinstance(last_transform, Compose):
            return ObjectCalculator.is_valid_transform(last_transform)
        else:
            return last_transform.ENT

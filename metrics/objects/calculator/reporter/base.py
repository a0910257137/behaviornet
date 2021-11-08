import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar


@dataclasses.dataclass
class Reporter(ABC):
    __default_frame_id__: ClassVar[str] = '__all_frame__'

    @abstractmethod
    def update(self):
        # Define how to update report based on metric results each time
        pass

    @property
    @abstractmethod
    def report(self):
        # Define reporter results
        pass

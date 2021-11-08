from abc import ABC, abstractmethod


class Metric(ABC):
    """define object to object metric
    """

    @abstractmethod
    def __call__(self):
        # object to object metric
        pass

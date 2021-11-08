from .accuracy import AccCalculator
from .precision_recall_f1score import PRFCalculator

CALCULATOR_FACTORY = {
    'PRFCalculator': PRFCalculator(),
    "AccCalculator": AccCalculator()
}

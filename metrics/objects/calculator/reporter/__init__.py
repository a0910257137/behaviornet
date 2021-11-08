from .accuracy import AccReporter
# from .base import Reporter
from .edge import EdgePRFReporter

REPORTER_FACTORY = {
    "EdgePRFReporter": EdgePRFReporter(),
    "AccReporter": AccReporter()
}

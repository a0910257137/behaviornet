from .head import Head

from .tood_head import TOODHead
# from .scrfd_sd_head import SCRFDHead
from .scrfd_head import SCRFDHead

HEAD_FACTORY = dict(
    head=Head,
    scrfd=SCRFDHead,
    # scrfd_sd=SCRFDHead,
    tood=TOODHead,
)

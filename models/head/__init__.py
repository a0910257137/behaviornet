from .head import Head
from .scrfd_head import SCRFDHead
from .tood_head import TOODHead

HEAD_FACTORY = dict(
    head=Head,
    scrfd=SCRFDHead,
    tood=TOODHead,
)

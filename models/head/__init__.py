from .head import Head
from .scrfd_head import SCRFDHead

HEAD_FACTORY = dict(
    head=Head,
    scrfd=SCRFDHead,
)

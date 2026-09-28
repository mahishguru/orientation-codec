"""
orientation_codec: HCP orientation encoding/decoding for image-based neural networks.

Encodes crystallographic orientations (Euler angles) from Dream3D files
into smooth RGB images suitable for VAE/ViT/Diffusion models, and decodes
generated RGB images back into Dream3D files.
"""

from orientation_codec.encoder import encode_dream3d
from orientation_codec.decoder import decode_rgb_to_dream3d
from orientation_codec.batch import batch_encode
from orientation_codec.dataset import (
    compute_class_means,
    load_class_means,
    encode_dream3d_global,
    decode_pixelwise,
    decode_image_pixelwise,
    decode_tiff_pixelwise,
    batch_encode_global,
    segment_grains,
)
from orientation_codec.io_utils import (
    read_dream3d,
    write_dream3d,
    save_png,
    load_png,
    save_16bit_tiff,
    load_16bit_tiff,
)
from orientation_codec.metrics import hcp_disorientation_deg, grain_disorientations, boundary_f1

__version__ = "0.1.0"
__all__ = [
    "encode_dream3d",
    "decode_rgb_to_dream3d",
    "batch_encode",
    "compute_class_means",
    "load_class_means",
    "encode_dream3d_global",
    "decode_pixelwise",
    "decode_image_pixelwise",
    "decode_tiff_pixelwise",
    "batch_encode_global",
    "segment_grains",
    "read_dream3d",
    "write_dream3d",
    "hcp_disorientation_deg",
    "grain_disorientations",
    "boundary_f1",
    "save_png",
    "load_png",
    "save_16bit_tiff",
    "load_16bit_tiff",
]

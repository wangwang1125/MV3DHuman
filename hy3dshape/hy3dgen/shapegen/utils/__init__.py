# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# Import from hy3dshape.utils to maintain compatibility
import sys
import os

# Add parent directory to path to import from hy3dshape
_current_dir = os.path.dirname(os.path.abspath(__file__))
_hy3dshape_dir = os.path.join(_current_dir, '../../../../hy3dshape')
if _hy3dshape_dir not in sys.path:
    sys.path.insert(0, _hy3dshape_dir)

# Import from parent utils.py first (for logger, synchronize_timer, smart_load_model)
from ..utils import logger, synchronize_timer, smart_load_model

# Import from hy3dshape.utils for other utilities
from hy3dshape.utils.misc import instantiate_from_config, get_obj_from_str

# Also import trainings submodule
import hy3dshape.utils.trainings
trainings = hy3dshape.utils.trainings

__all__ = ['instantiate_from_config', 'get_obj_from_str', 'smart_load_model', 'trainings', 'logger', 'synchronize_timer']

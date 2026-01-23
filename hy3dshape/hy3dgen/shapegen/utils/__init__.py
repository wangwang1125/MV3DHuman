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

# Import from parent utils.py first (for logger, synchronize_timer, smart_load_model)
# Use direct file import to avoid circular import
import sys
import os
import importlib.util

# Get parent utils.py path (one level up from utils/ directory)
_parent_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils.py')

# Load parent utils.py as a module
_spec = importlib.util.spec_from_file_location("_hy3dgen_shapegen_utils", _parent_utils_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load utils.py from {_parent_utils_path}")
_parent_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent_utils_module)

# Import from parent utils.py
logger = _parent_utils_module.logger
synchronize_timer = _parent_utils_module.synchronize_timer
smart_load_model = _parent_utils_module.smart_load_model

# Import from hy3dshape.utils for other utilities
_current_dir = os.path.dirname(os.path.abspath(__file__))
_hy3dshape_dir = os.path.join(_current_dir, '../../../../hy3dshape')
if _hy3dshape_dir not in sys.path:
    sys.path.insert(0, _hy3dshape_dir)

from hy3dshape.utils.misc import instantiate_from_config, get_obj_from_str

# trainings submodule is now available as hy3dgen.shapegen.utils.trainings
# (created as a separate package in utils/trainings/__init__.py)

__all__ = ['instantiate_from_config', 'get_obj_from_str', 'smart_load_model', 'logger', 'synchronize_timer']

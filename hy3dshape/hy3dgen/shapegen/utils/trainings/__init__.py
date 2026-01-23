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

# Import all from hy3dshape.utils.trainings to maintain compatibility
import sys
import os

# Add parent directory to path to import from hy3dshape
_current_dir = os.path.dirname(os.path.abspath(__file__))
_hy3dshape_dir = os.path.join(_current_dir, '../../../../../hy3dshape')
if _hy3dshape_dir not in sys.path:
    sys.path.insert(0, _hy3dshape_dir)

# Import all from hy3dshape.utils.trainings
# Make submodules available first
import hy3dshape.utils.trainings.lr_scheduler
import hy3dshape.utils.trainings.mesh_log_callback
import hy3dshape.utils.trainings.mesh
import hy3dshape.utils.trainings.peft
import hy3dshape.utils.trainings.callback

# Register submodules in sys.modules so they can be imported as hy3dgen.shapegen.utils.trainings.xxx
# This allows imports like: from hy3dgen.shapegen.utils.trainings.mesh_log_callback import ...
sys.modules['hy3dgen.shapegen.utils.trainings.lr_scheduler'] = hy3dshape.utils.trainings.lr_scheduler
sys.modules['hy3dgen.shapegen.utils.trainings.mesh_log_callback'] = hy3dshape.utils.trainings.mesh_log_callback
sys.modules['hy3dgen.shapegen.utils.trainings.mesh'] = hy3dshape.utils.trainings.mesh
sys.modules['hy3dgen.shapegen.utils.trainings.peft'] = hy3dshape.utils.trainings.peft
sys.modules['hy3dgen.shapegen.utils.trainings.callback'] = hy3dshape.utils.trainings.callback

# Import specific classes
from hy3dshape.utils.trainings.lr_scheduler import LambdaWarmUpCosineFactorScheduler
from hy3dshape.utils.trainings.mesh_log_callback import (
    ImageConditionalASLDiffuserLogger,
    ImageConditionalFixASLDiffuserLogger
)

# Make submodules available as attributes
lr_scheduler = hy3dshape.utils.trainings.lr_scheduler
mesh_log_callback = hy3dshape.utils.trainings.mesh_log_callback
mesh = hy3dshape.utils.trainings.mesh
peft = hy3dshape.utils.trainings.peft
callback = hy3dshape.utils.trainings.callback

__all__ = [
    'LambdaWarmUpCosineFactorScheduler',
    'ImageConditionalASLDiffuserLogger',
    'ImageConditionalFixASLDiffuserLogger',
    'lr_scheduler',
    'mesh_log_callback',
    'mesh',
    'peft',
    'callback',
]

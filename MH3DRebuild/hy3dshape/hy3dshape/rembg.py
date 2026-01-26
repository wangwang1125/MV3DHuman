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

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    def __init__(self, model_name='u2net'):
        """
        初始化背景移除器
        
        Args:
            model_name (str): rembg模型名称，可选值包括：
                - 'u2net': 默认模型，通用场景
                - 'u2netp': 轻量级模型，速度更快
                - 'u2net_human_seg': 专门用于人物分割
                - 'silueta': Silueta模型
                - 'isnet-general-use': ISNet通用模型
                - 'sam': Segment Anything Model (需要额外配置)
        """
        self.model_name = model_name
        self.session = new_session(model_name=model_name)

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output

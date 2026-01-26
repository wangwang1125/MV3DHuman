'''
MH3DRebuild: Multi-view RGB 3D reconstruction with MV LoRA.
'''


class IOData:
    id = -1
    name = ""
    type = ""
    bref = ""
    sf = False  # self 为 python 保留字，防止歧义

    def __init__(self, id, name, type, bref, sf):
        self.id = id
        self.name = name
        self.type = type
        self.bref = bref
        self.sf = sf


class Info:
    SOFTWARE_AUTHOR = "MH3DRebuild"
    SOFTWARE_ATTENTION = "Hunyuan3D MV LoRA 3D Reconstruction."
    SOFTWARE_NAME = "MH3DRebuild"
    SOFTWARE_NAME_ZH = "多视图RGB三维重建"
    SOFTWARE_ABBR = "MH3DR"
    SOFTWARE_VERSION = "v1.0.0"
    SOFTWARE_BRIEF = "基于四视图RGB与MV LoRA微调模型的三维mesh重建"
    SOFTWARE_WITH_CUDA = 1

    def __init__(self):
        self.Input = []
        self.Input.append(
            IOData(0, "input_data", "DIR", "四视图图像目录，含 front/left/back/right 对应图像", False)
        )
        self.Input.append(
            IOData(1, "models", "DIR", "可选；LoRA checkpoint 或 PEFT 目录", True)
        )

        self.Output = []
        self.Output.append(
            IOData(0, "output_mesh", "GLB", "重建得到的 mesh 文件路径", False)
        )

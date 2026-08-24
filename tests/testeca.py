import unittest
import torchvision.models as models
from mobilenetv2_eca.ECANet.models import ECA_MobileNetV2, eca_layer

# 无 ECA 的 MobileNetV2
base_model = models.mobilenet_v2()
print(f"Base Params: {sum(p.numel() for p in base_model.parameters())}")

# 有 ECA 的 MobileNetV2
eca_model = ECA_MobileNetV2()
print(f"ECA Params: {sum(p.numel() for p in eca_model.parameters())}")
"""Model factory: timm backbones plus the scratch CNN (PLAN.md section 6)."""
import timm
import torch.nn as nn

# Short names used on the command line -> timm names (PLAN.md section 7)
ALIASES = {
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "vgg16_bn": "vgg16_bn.tv_in1k",
    "effnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
}


class ScratchCNN(nn.Module):
    """M1: 5 blocks of (Conv-BN-ReLU) x2 + MaxPool, GAP, dropout, FC -> 2."""

    def __init__(self, num_classes=2, widths=(32, 64, 128, 256, 256), dropout=0.3, init="kaiming"):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            for _ in range(2):
                layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
                c_in = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))
        if init == "kaiming":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    def get_classifier(self):
        return self.head


def build_model(name, pretrained=True, img_size=None, init="kaiming"):
    """Returns (model, data_cfg, is_vit). data_cfg has input_size, mean, std, crop_pct, interpolation."""
    if name == "scratch_cnn":
        size = img_size or 128
        model = ScratchCNN(init=init)
        cfg = {"input_size": (3, size, size), "crop_pct": 0.875, "interpolation": "bilinear"}
        return model, cfg, False

    timm_name = ALIASES.get(name, name)
    kwargs = {"num_classes": 2}
    is_vit = timm_name.startswith("vit_")
    if is_vit:
        kwargs["img_size"] = img_size or 224   # DINOv2 defaults to 518; timm interpolates pos-embeds
    model = timm.create_model(timm_name, pretrained=pretrained, **kwargs)
    head = model.get_classifier()
    std = head.weight.std().item()
    if std > 0.05:   # timm's EfficientNet inits a 2-class head at std ~0.41; bring it in line with the others
        nn.init.trunc_normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        print(f"re-initialised head (std {std:.2f} -> 0.02)", flush=True)
    cfg = dict(timm.data.resolve_data_config({}, model=model))
    if is_vit:
        cfg["input_size"] = (3, kwargs["img_size"], kwargs["img_size"])   # pretrained_cfg still says 518
    elif img_size:
        cfg["input_size"] = (3, img_size, img_size)
    return model, cfg, is_vit


def head_parameters(model):
    return list(model.get_classifier().parameters())

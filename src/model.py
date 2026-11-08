"""Classifier definition. All experiments share one architecture; only the training data changes."""
import torch
import torchvision


def build_model(cfg, pretrained=True):
    num_classes = len(cfg["data"]["classes"])
    name = cfg["train"]["backbone"]
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model


def load_trained(cfg, path, device):
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()

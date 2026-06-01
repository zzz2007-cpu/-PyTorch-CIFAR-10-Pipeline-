from models.resnet18 import build_resnet18
from models.simple_cnn import SimpleCNN


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")

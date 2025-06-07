# xai_methods/utils.py

import torch
import torchvision.transforms as T
from PIL import Image

def load_model(arch: str, model_path=None):
    from torchvision.models import resnet50, densenet121
    model = None
    if arch == "resnet50":
        model = resnet50(pretrained=(model_path is None))
    elif arch == "densenet121":
        model = densenet121(pretrained=(model_path is None))
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    if model_path:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

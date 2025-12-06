# src.preprocess.py
import torch
from torchvision import transforms
from PIL import Image

class CLIPPreprocessor:
    """
    Provides:
    - image preprocessing, (resize,crop,normalize)
    - text preprocessing, (lowercase, strip)
    Ensures deterministic processing across training and inference.
    """
    def __init__(self,image_size=224):
        self.image_size = image_size

        # Image Preprocessing Pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size,interpolation=Image.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    def preprocess_image(self ,image_path):
        # Loads an image, applies a transform, returns a tensor.
        image = Image.open(image_path).convert('RGB')
        return self.image_transform(image)


    def preprocess_text(self,text:str):
        """
        Applies simple normalization for text embeddings
        """
        return text.strip().lower()
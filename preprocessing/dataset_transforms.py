import torch
import torchvision.transforms.v2 as transforms

def get_classification_transforms():
    """Retorna una composición de transformaciones para clasificación."""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Redimensionar la imagen a 256x256
        transforms.RandomCrop(224),     # Recorte aleatorio de 224x224
        transforms.RandomRotation(30),  # Rotación aleatoria de hasta 30 grados
        transforms.ToTensor(),          # Convertir la imagen a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalización
    ])

def get_segmentation_transforms():
    """Retorna una composición de transformaciones para segmentación."""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Redimensionar la imagen a 256x256
        transforms.CenterCrop(224),     # Recorte central de 224x224
        transforms.ToTensor(),          # Convertir la imagen a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalización
    ])


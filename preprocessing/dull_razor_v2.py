import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image


def dull_razor(image, kernel_size=17):
    """
    Aplica el algoritmo DullRazor para remover pelos de imágenes dermatológicas.

    Args:
    - image (np.array): Imagen de entrada en formato de array numpy.
    - kernel_size (int): Tamaño del kernel para la convolución. Valor por defecto es 17.

    Returns:
    - np.array: Imagen procesada con el algoritmo DullRazor.
    """
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray_scale, 3)
    edges = cv2.Canny(median, 10, 60)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    filtered = cv2.inpaint(
        image, dilated_edges, inpaintRadius=1, flags=cv2.INPAINT_TELEA
    )
    return filtered


class DullRazorTransform:
    def __init__(self, kernel_size=17):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img_array = np.array(img)
        img_array = dull_razor(img_array, self.kernel_size)
        return Image.fromarray(img_array)


def get_classification_transforms_with_dull_razor(kernel_size=17):
    """Retorna una composición de transformaciones para clasificación incluyendo DullRazor."""
    return transforms.Compose(
        [
            DullRazorTransform(kernel_size),  # Aplicar DullRazor
            transforms.Resize((256, 256)),  # Redimensionar la imagen a 256x256
            transforms.RandomCrop(224),  # Recorte aleatorio de 224x224
            transforms.RandomRotation(30),  # Rotación aleatoria de hasta 30 grados
            transforms.ToTensor(),  # Convertir la imagen a tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalización
        ]
    )


def get_segmentation_transforms_with_dull_razor(kernel_size=17):
    """Retorna una composición de transformaciones para segmentación incluyendo DullRazor."""
    return transforms.Compose(
        [
            DullRazorTransform(kernel_size),  # Aplicar DullRazor
            transforms.Resize((256, 256)),  # Redimensionar la imagen a 256x256
            transforms.CenterCrop(224),  # Recorte central de 224x224
            transforms.ToTensor(),  # Convertir la imagen a tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalización
        ]
    )

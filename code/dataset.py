import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def crop_face(image, landmarks, padding=50):
    """
    Recadre une image autour du visage en utilisant les coordonnées maximales et minimales des points de repère du visage.
    
    Args:
    - image (PIL Image): L'image à recadrer.
    - landmarks (list or np.array): Un vecteur de 136 éléments contenant les coordonnées x et y des 68 points de repère.
    - padding (int): Le padding en pixels à ajouter autour du rectangle englobant. Par défaut : 10px.
    
    Returns:
    - cropped_image (PIL Image): L'image recadrée autour du visage.
    """
    # Séparer les coordonnées x et y du vecteur
    x_coords = np.array(landmarks[:68])  # Les 68 premières valeurs correspondent aux x
    y_coords = np.array(landmarks[68:])  # Les 68 dernières valeurs correspondent aux y

    # Obtenir les coordonnées minimales et maximales
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Ajouter un padding et s'assurer que les coordonnées ne dépassent pas les dimensions de l'image
    x_min = max(0, int(x_min - padding))
    y_min = max(0, int(y_min - padding))
    x_max = min(image.width, int(x_max + padding))  # Utiliser image.width pour une image PIL
    y_max = min(image.height, int(y_max + padding))  # Utiliser image.height pour une image PIL

    # Recadrer l'image
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    return cropped_image

class Emotion_Dataset(Dataset):
    def __init__(self, X, labels, img_dir, transform=None):
        '''
        Pytorch Dataset class
        params:-
                 X       : list containing image paths and facial landmark points
                 labels  : list of labels corresponding to the images
                 img_dir : the directory of the images
                 transform: pytorch transformations to be applied on the data
        return :-
                 image, labels
        '''
        self.X = X
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract the image path and the landmarks from X
        img_path = self.img_dir + f"/{self.X[idx][0]}"
        landmarks = np.array(self.X[idx][2:], dtype=np.float32)  # Extract landmarks
        
        # Load image
        img = Image.open(img_path).convert("RGB")

        # img = draw_landmarks(img, landmarks)

        img = crop_face(img, landmarks, padding=30)
        # Draw landmarks on the image
        # img_with_landmarks = self.draw_landmarks(img, landmarks)

        # Apply transformations if provided
        if self.transform:
            img_with_landmarks = self.transform(img)

        # Convert label to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_with_landmarks, label
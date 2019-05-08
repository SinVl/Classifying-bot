from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from fastai.vision import load_learner, Image, open_image
import numpy as np

class ClassPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_learner("../model/data", fname="trained_model.pkl")
        self.to_tensor = transforms.ToTensor()

    def predict(self, img_path):

        pred  = self.model.predict(self.process_image(img_path))
        class_ = pred[0]
        probability = np.max(np.array(pred[2]))*100
        return class_,probability
    
    def process_image(self, img_path):
        img = open_image(img_path)
        return img

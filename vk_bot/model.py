from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from fastai.vision import load_learner, Image, open_image
import numpy as np

# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class ClassPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_learner("../model/data", fname="trained_model.pkl")
        self.to_tensor = transforms.ToTensor()

    def predict(self, img_path):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # Не забудьте перенести все трансофрмации, которые вы использовали при тренировке
        # Для этого будет удобно сохранить питоновский объект с ними в виде файла с помощью pickle,
        # а потом загрузить здесь.
        
        # Обработка картинки сейчас производится в методе process image, а здесь мы должны уже применить нашу
        # модель и вернуть вектор предсказаний для нашей картинки

        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        pred  = self.model.predict(self.process_image(img_path))
        class_ = pred[0]
        probability = np.max(np.array(pred[2]))*100
        return class_,probability

    # В predict используются некоторые внешние функции, их можно добавить как функции класса
    # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
    # принято ставить _ (выглядит это так: def _foo() )
    # ниже пример того, как переносить методы
    def process_image(self, img_path):
        # используем PIL, чтобы получить картинку из потока и изменить размер
        #image = PIL_Image.open(img_stream).resize((256, 256))
        # переводим картинку в тензор и оборачиваем в объект Image, который использует fastai у себя внутри
        #image = Image(self.to_tensor(image))
        img = open_image(img_path)
        return img

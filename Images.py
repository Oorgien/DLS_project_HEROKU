import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

class Preproc(nn.Module):
    def __init__(self, imsize):
        super(Preproc, self).__init__()
        self.imsize = imsize
        self.loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат

        self.unloader = transforms.ToPILImage()  # тензор в кратинку

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image

    def transform_images(self, content, style_1, style_2):
        style1_img = self.image_loader(style_1)  # as well as here
        style2_img = self.image_loader(style_2)
        content_img = self.image_loader(content)  # измените путь на тот который у вас.
        return style1_img, style2_img, content_img


class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std


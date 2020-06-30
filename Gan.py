from torch import *

import torch
import torch.nn as nn

from Images import Preproc, Normalization

class GanModel(nn.Module):
    normalization_mean = torch.tensor([0.5, 0.5, 0.5])
    normalization_std = torch.tensor([0.5, 0.5, 0.5])

    def __init__(self, model_name):
        super(GanModel, self).__init__()
        self.model_jit_name = './pytorch_CycleGAN_and_pix2pix/checkpoints/' + model_name + '/latest_net_G.jit'
        self.model = self.load_model()

    def load_model(self):
        model_file = self.model_jit_name
        model = torch.jit.load(model_file)
        model = model.eval()
        return model

    def forward(self, img, img_size):

        normalization = Normalization(self.normalization_mean, self.normalization_std)
        content_img = Preproc(img_size).image_loader(img)
        normalized_img = normalization.forward(content_img)

        with torch.no_grad():
            res = self.model(normalized_img)
        res = res.view(res.shape[1], res.shape[2], res.shape[3])
        res = (res * self.normalization_std.view(3, 1, 1)) + self.normalization_mean.view(3, 1, 1)
        return res
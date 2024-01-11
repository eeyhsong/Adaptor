"""
DOMAIN ADAPTION Transformer for EEG Classification

Multi-branch + transformer (*cross attention*) + adversarial learning + adaptive center loss
Basic Version of the paper
------
Use conformer as the backbone.
Significant improvement than v0.65!! 

Use the validation set to find the best model (with the best val_acc or val_loss)
"""


import argparse
import os
gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

log_path = './results/best_val_acc/'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.cuda.FloatTensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# TODO: Encoder is for source data, Decoder is for target data
class Feature_Extractor_Enc(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()

        self.temporal_spatial = nn.Sequential(
            nn.Conv2d(1, 20, (1, 25), (1, 1)),
            # nn.BatchNorm2d(40),
            # nn.ELU(),
            # nn.Dropout(0.3),
            nn.Conv2d(20, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(1, 20, (22, 1), (1, 1)),
            # nn.BatchNorm2d(40),
            # nn.ELU(),
            # nn.Dropout(0.3),
            nn.Conv2d(20, 40, (1, 25), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, X: Tensor) -> Tensor:
        x, y = X[0], X[1]
        x = self.temporal_spatial(x) + self.spatial_temporal(x)
        x = self.projection(x)

        y = self.temporal_spatial(y) + self.spatial_temporal(y)
        y = self.projection(y)
        return (x, y)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class MultiHeadAttention_Dec(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x = X[1] # target data
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class MultiHeadAttention_Enc_Dec(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, X, mask: Tensor = None) -> Tensor:
        x_enc, x_dec = X[0], X[1] # enc is target, dec is source
        queries = rearrange(self.queries(x_dec), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x_dec), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x_enc), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = y
        y = self.fn(y, **kwargs)
        y += res
        return (x, y)

class ResidualAdd_src(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return (x, y)


class ResidualAdd_Dec1(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return (x, y)

class ResidualAdd_Dec2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

        self.lm = nn.LayerNorm(40)

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.lm(x)
        x = self.fn((x, y), **kwargs)
        x += res
        return (x, y)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerDecoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd_Dec1(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention_Dec(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd_Dec2(nn.Sequential(
                MultiHeadAttention_Enc_Dec(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd_Dec1(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerBlock(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__(
            TransformerEncoderBlock(emb_size),
            TransformerDecoderBlock(emb_size)
        )


class Transformer(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerBlock(emb_size) for _ in range(depth)])


class Feature_Extractor(nn.Sequential):
    def __init__(self, emb_size=40, depth=3):
        super().__init__(
            Feature_Extractor_Enc(emb_size),
            Transformer(depth, emb_size)
        )


class Classifier(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=4):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class Discriminator(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=2, **kwargs):
        super().__init__(
        )
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = self.clshead(x)
        return x


class DATrans():
    def __init__(self, nsub):
        super(DATrans, self).__init__()
        self.batch_size = 64
        self.n_epochs = 1000
        self.c_dim = 4
        self.lr = 0.0002  # original 0.0001
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (61, 40)  # (475, 20)
        self.lambda_cen = 0.5
        self.lambda_cls = 2
        # self.lambda_cls_irr = 0.5
        self.lambda_gp = 10
        self.alpha = 0.0002
        self.nSub = nsub

        self.root = './data/strict_TE/'

        self.log_write = open(log_path + "log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.Feature_Extractor = nn.DataParallel(Feature_Extractor()).cuda()
        self.Classifier = nn.DataParallel(Classifier()).cuda()
        self.Discriminator = nn.DataParallel(Discriminator()).cuda()

        self.centers = {}

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 8), 1, 22, 1000))
            for ri in range(int(self.batch_size / 8)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 8)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_data(self):  # get source and target data
        
        source_data = []
        source_label = []
        # to get the data of source subject
        for sub_index in range(9):
            sub_index += 1
            if sub_index != self.nSub:
                tmp = scipy.io.loadmat(self.root + 'A0%dT.mat' % sub_index)
                tmp_one_sub_data = tmp['data']
                tmp_one_sub_label = tmp['label']

                tmp_one_sub_data = np.transpose(tmp_one_sub_data, (2, 1, 0))
                tmp_one_sub_data = np.expand_dims(tmp_one_sub_data, axis=1)
                tmp_one_sub_label = np.transpose(tmp_one_sub_label)

                tmp_one_sub_label = tmp_one_sub_label[0]
                source_data.append(tmp_one_sub_data)
                source_label.append(tmp_one_sub_label)
        self.source_data = np.concatenate(source_data)
        self.source_label = np.concatenate(source_label)
        
        # shuffle
        shuffle_num = np.random.permutation(len(self.source_data))
        self.source_data = self.source_data[shuffle_num, :, :, :]
        self.source_label = self.source_label[shuffle_num]

        # to get the data of target subject
        self.target_tmp = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.target_tmp['data']
        self.train_label = self.target_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.target_data = self.train_data
        self.target_label = self.train_label[0]

        # shuffle target data 
        shuffle_num = np.random.permutation(len(self.target_data))
        self.target_data = self.target_data[shuffle_num, :, :, :]
        self.target_label = self.target_label[shuffle_num]

        # val set
        self.val_data = self.target_data[:32]
        self.val_label = self.target_label[:32]

        # target set
        self.target_data = self.target_data[32:]
        self.target_label = self.target_label[32:]


        # correspond to the number of source data
        tmp_d = self.target_data
        tmp_l = self.target_label

        self.full_data = np.concatenate([tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d])
        self.full_label = np.concatenate([tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l])


        # test data
        # to get the data of target subject
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.test_data = self.test_data
        self.test_label = self.test_label[0]


        return self.source_data, self.source_label, self.full_data, self.full_label, self.val_data, self.val_label, self.test_data, self.test_label


    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def update_centers(self, feature, label):
            deltac = {}
            count = {}
            count[0] = 0
            for i in range(len(label)):
                l = label[i]
                if l in deltac:
                    deltac[l] += self.centers[l]-feature[i]
                else:
                    deltac[l] = self.centers[l]-feature[i]
                if l in count:
                    count[l] += 1
                else:
                    count[l] = 1

            for ke in deltac.keys():
                deltac[ke] = deltac[ke]/(count[ke]+1)

            return deltac

    def train(self):

        self.Feature_Extractor.apply(weights_init_normal)
        self.Classifier.apply(weights_init_normal)
        self.Discriminator.apply(weights_init_normal)

        sour_img, sour_label, img, label, val_data, val_label, test_data, test_label = self.get_data()

        sour_shuflle_num = np.random.permutation(len(sour_img))
        sour_img = sour_img[sour_shuflle_num, :, :, :]
        sour_label = sour_label[sour_shuflle_num]

        sour_img = torch.from_numpy(sour_img)
        sour_label = torch.from_numpy(sour_label - 1)
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        val_data = torch.from_numpy(val_data)
        val_label = torch.from_numpy(val_label - 1)


        dataset = torch.utils.data.TensorDataset(img, label, sour_img, sour_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(self.c_dim):
            self.centers[i] = torch.randn(self.dimension)
            self.centers[i] = self.centers[i].cuda()

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Feature_Extractor.parameters(), self.Classifier.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_dis = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        bestAcc = 0
        averAcc = 0
        num = 0
        gamma = 1
        best_acc_val = 0

        for e in range(self.n_epochs):
            tacc = 0
            tnum = 0
            self.Feature_Extractor.train()
            self.Classifier.train()
            self.Discriminator.train()

            for i, (img, label, sour_img, sour_label) in enumerate(self.dataloader):

                img = Variable(img.type(self.Tensor))
                label = Variable(label.type(self.LongTensor))
                sour_img = Variable(sour_img.type(self.Tensor))
                sour_label = Variable(sour_label.type(self.LongTensor))

                # --------------
                #  Train the domain discriminator
                # --------------

                # if i > 20 & (i + 1) % 1 == 0:
                if (i + 1) % 1 == 0:
                    self.optimizer_dis.zero_grad()

                    (sour_feature, feature) = self.Feature_Extractor((sour_img, img))

                    # discriminator
                    pre_dom = self.Discriminator(feature.detach())
                    pre_dom_sour = self.Discriminator(sour_feature.detach())


                    # Adversarial loss
                    gradient_penalty = compute_gradient_penalty(self.Discriminator, feature, sour_feature)
                    loss_D_GAN = - torch.mean(pre_dom) + torch.mean(pre_dom_sour) + self.lambda_gp * gradient_penalty

                    loss_D = loss_D_GAN

                    loss_D.backward()
                    self.optimizer_dis.step()

                # --------------
                #  Train the united networks, including the encoder and the classifier
                # --------------
                if (i + 1) % 1 == 0:
                    self.optimizer.zero_grad()

                    aug_data, aug_label = self.interaug(self.target_data, self.target_label)
                    img = torch.cat((img[:32], aug_data))
                    label = torch.cat((label[:32], aug_label))

                    (sour_feature, feature) = self.Feature_Extractor((sour_img, img))
                    # classifier
                    out_cls = self.Classifier(feature)
                    sour_out_cls = self.Classifier(sour_feature)
                    # discriminator
                    pre_cls_fake = self.Discriminator(sour_feature)

                    # Classification loss
                    loss_cls_targ = self.criterion_cls(out_cls, label)
                    # writer.add_scalar('Joint/cls_target', loss_cls_targ, e)
                    # writer.flush()

                    loss_cls_sour = self.criterion_cls(sour_out_cls, sour_label)
                    # writer.add_scalar('Joint/cls_source', loss_cls_sour, e)
                    # writer.flush()

                    loss_Joint_cls = loss_cls_targ + loss_cls_sour

                    # Training accuracy for target data
                    for tk in range(len(label)):
                        tnum = tnum + 1
                        train_pred = torch.max(out_cls, 1)[1]
                        if train_pred[tk] == label[tk]:
                            tacc = tacc + 1


                    # Adversarial loss
                    loss_Joint_adv = - torch.mean(pre_cls_fake)

                    # Central loss
                    cen_feature_st = torch.cat((feature, sour_feature), axis=0)  # source and target
                    cen_label_st = torch.cat((label, sour_label))

                    cen_feature = feature
                    cen_label = label
                    nplabela = cen_label_st.cpu().numpy()

                    # Center loss
                    loss_Cen = 0
                    for k in range(len(cen_label_st)):
                        la = nplabela[k]
                        if k == 0:
                            loss_Cen = self.criterion_l2(self.centers[la], cen_feature_st[k])
                        else:
                            loss_Cen += self.criterion_l2(self.centers[la], cen_feature_st[k])
                    # writer.add_scalar('Joint/cen', loss_Cen, e)
                    # writer.flush()


                    loss_U = loss_Joint_cls + loss_Joint_adv + self.lambda_cen/5 * loss_Cen  
                    # writer.add_scalar('Total/Joint', loss_U, e)
                    # writer.flush()

                    loss_U.backward()
                    self.optimizer.step()

                    # update centers
                    deltacA = self.update_centers(cen_feature, cen_label.cpu().numpy())

                    with torch.no_grad():
                        for ke in deltacA.keys():
                            self.centers[ke] = self.centers[ke] - self.alpha * deltacA[ke]

            tacc = 1.0 * tacc / tnum

            if (e + 1) % 1 == 0:
                self.Feature_Extractor.eval()
                self.Classifier.eval()
                # self.Discriminator.eval()

                with torch.no_grad():
                    vacc = 0
                    vnum = 0

                    val_data = Variable(val_data.type(self.Tensor))
                    val_label = Variable(val_label.type(self.LongTensor))

                    (_, vfeature) = self.Feature_Extractor((val_data, val_data))
                    vCls = self.Classifier(vfeature)

                    y_pred = torch.max(vCls, 1)[1]

                    loss_cls_val = self.criterion_cls(vCls, val_label)

                    for k in range(len(val_label)):
                        vnum = vnum + 1
                        if y_pred[k] == val_label[k]:
                            vacc = vacc + 1
                    vacc = 1.0 * vacc / vnum

                    if vacc > best_acc_val:
                        best_acc_val = vacc
                        best_epoch = e
                        torch.save(self.Feature_Extractor.state_dict(), "model/sub%d_Enc_ac1.pth" % self.nSub)
                        torch.save(self.Classifier.state_dict(), "model/sub%d_Cls_ac1.pth" % self.nSub)
                        torch.save(self.Discriminator.state_dict(), "model/sub%d_Dis_ac1.pth" % self.nSub)

            print(
                'Epoch: %d  Loss_D: %.4f  Loss_G: %.4f  Loss_cls_sour: %.4f  Loss_cls_targ: %.4f  Loss_cen: %.4f  Train_acc: %.5f  Val_loss: %.5f  Val_acc: %.5f'
                % (e, loss_D, loss_Joint_adv, loss_cls_sour, loss_cls_targ, loss_Cen, tacc, loss_cls_val, vacc))

        self.Feature_Extractor.load_state_dict(torch.load("model/sub%d_Enc_ac1.pth" % self.nSub))
        self.Classifier.load_state_dict(torch.load("model/sub%d_Cls_ac1.pth" % self.nSub))
        self.Feature_Extractor.eval()
        self.Classifier.eval()
        
        with torch.no_grad():

            acc = 0
            num = 0

            test_data = Variable(test_data.type(self.Tensor))
            test_label = Variable(test_label.type(self.LongTensor))

            (feature, feature) = self.Feature_Extractor((test_data, test_data))
            Cls = self.Classifier(feature)

            y_pred = torch.max(Cls, 1)[1]

            for k in range(len(test_label)):
                num = num + 1
                if y_pred[k] == test_label[k]:
                    acc = acc + 1
            acc = 1.0 * acc / num
            self.log_write.write(str(e) + "    " + str(acc) + "\n")


        averAcc = averAcc / num
        print('The best epoch is:', best_epoch)
        print('The test accuracy is:', acc)
        self.log_write.write('The best epoch is: ' + str(best_epoch) + "\n")
        self.log_write.write('The test accuracy is: ' + str(acc) + "\n")

        return best_epoch, acc
        # writer.close()


def main():
    best = 0
    aver = 0
    result_write = open(log_path + "sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2022)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        datrans = DATrans(i + 1)
        best_epoch, acc = datrans.train()
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best epoch is: ' + str(best_epoch) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(acc) + "\n")
        # best = best + bestAcc
        aver = aver + acc
        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))

    best = best / 9
    aver = aver / 9
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))


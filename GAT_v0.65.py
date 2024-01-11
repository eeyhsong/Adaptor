"""
DOMAIN ADAPTION Transformer for EEG Classification

Multi-branch + transformer (*cross attention*) + adversarial learning + adaptive center loss

Basic Version of the paper
"""


import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn.init as init
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


class preEncoder_Sour(nn.Module):
    def __init__(self, emb_size=50):
        super().__init__()
        self.temporal1 = nn.Sequential(
            nn.Conv2d(1, 10, (1, 51), stride=(1, 1), padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.spatial1 = nn.Sequential(
            nn.Conv2d(10, 10, (22, 1), (1, 1)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.temporal2 = nn.Sequential(
            nn.Conv2d(10, 10, (1, 51), stride=(1, 1), padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(1, 10, (22, 1), (1, 1)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(10, emb_size, (1, 5), stride=(1, 5)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, X) -> Tensor:
        x, y = X[0], X[1]
        x = self.spatial1(self.temporal1(x)) + self.temporal2(self.spatial2(x))
        x = self.projection(x)
        return (x, y)


class preEncoder_Targ(nn.Module):
    def __init__(self, emb_size=50):
        super().__init__()
        self.temporal1 = nn.Sequential(
            nn.Conv2d(1, 10, (1, 51), stride=(1, 1), padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.spatial1 = nn.Sequential(
            nn.Conv2d(10, 10, (22, 1), (1, 1)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.temporal2 = nn.Sequential(
            nn.Conv2d(10, 10, (1, 51), stride=(1, 1), padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(1, 10, (22, 1), (1, 1)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(10, emb_size, (22, 2), stride=(1, 2)),  # 5 is better than 1
            # nn.MaxPool2d( kernel_size=(1,5), stride=(1,5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
            # nn.LayerNorm()
        )

        self.projection_test = nn.Sequential(
            # nn.MaxPool2d((1, 50), (1, 15)),
            nn.Conv2d(10, emb_size, (1, 5), stride=(1, 5)),  # 5 is better than 1
            # nn.MaxPool2d( kernel_size=(1,5), stride=(1,5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
            # nn.LayerNorm()
        )

    def forward(self, X) -> Tensor:
        x, y = X[0], X[1]
        y = self.spatial1(self.temporal1(y)) + self.temporal2(self.spatial2(y))
        y = self.projection_test(y)
        return (x, y)


class MultiHeadAttention_Enc(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, X: Tensor, mask: Tensor = None) -> Tensor:
        x = X[1] # target data 
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

    def forward(self, X: Tensor, mask: Tensor = None) -> Tensor:
        x = X[0] # source data 
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


# It's a little confused, use TransEnc for target data and TransDec for source data.
# But the input x is source, y is target

class TransformerDecoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            # ResidualAdd_Dec1(nn.Sequential(
            #     nn.LayerNorm(emb_size),
            #     MultiHeadAttention_Dec(emb_size, num_heads, drop_p),
            #     nn.Dropout(drop_p)
            # )),
            ResidualAdd_Dec2(nn.Sequential(
                # nn.LayerNorm(emb_size),
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
            TransformerDecoderBlock(emb_size)
        )


class Transformer(nn.Sequential):
    def __init__(self, depth, emb_size=50):
        super().__init__(*[TransformerBlock(emb_size) for _ in range(depth)])


class Encoder(nn.Sequential):
    def __init__(self, emb_size=50, depth=3):
        super().__init__(
            preEncoder_Sour(emb_size),
            preEncoder_Targ(emb_size)
        )


class Feature_Extractor(nn.Sequential):
    def __init__(self, emb_size=50, depth=1, **kwargs):
        super().__init__(
            Encoder(emb_size),
            Transformer(depth, emb_size),
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(190, 1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(600, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class Classifier(nn.Sequential):
    def __init__(self, emb_size=50, depth=3, n_classes=4, **kwargs):
        super().__init__(
            ClassificationHead(emb_size, n_classes)
        )


class Discriminator(nn.Sequential):
    def __init__(self, emb_size=50, depth=3, n_classes=2, **kwargs):
        super().__init__(
        )
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(600, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = self.clshead(x)
        return x


class DATrans():
    def __init__(self, nsub):
        super(DATrans, self).__init__()
        self.batch_size = 64
        self.n_epochs = 2000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002  # original 0.0001
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)  # (475, 20)
        self.lambda_cen = 0.5
        self.lambda_cls = 2
        self.lambda_cls_irr = 0.5
        self.lambda_gp = 10
        self.alpha = 0.0002
        self.nSub = nsub

        self.start_epoch = 0
        self.root = './data/standard_2a_data/strict_TE/'

        self.pretrain = False

        self.log_write = open("./results/test/log_subject%d.txt" % self.nSub, "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.Feature_Extractor = Feature_Extractor()
        self.Classifier = Classifier()
        self.Discriminator = Discriminator()


        self.Feature_Extractor = nn.DataParallel(self.Feature_Extractor)
        self.Classifier = nn.DataParallel(self.Classifier)
        self.Discriminator = nn.DataParallel(self.Discriminator)

        self.Feature_Extractor = self.Feature_Extractor.cuda()
        self.Classifier = self.Classifier.cuda()
        self.Discriminator = self.Discriminator.cuda()

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
        def aug(img, label):
            aug_data = []
            aug_label = []
            for cls4aug in range(4):
                cls_idx = np.where(label == cls4aug + 1)
                tmp_data = img[cls_idx]
                tmp_label = label[cls_idx]

                tmp_aug_data = np.zeros(tmp_data.shape)
                for ri in range(tmp_data.shape[0]):
                    for rj in range(8):
                        rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                        tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                          rj * 125:(rj + 1) * 125]

                aug_data.append(tmp_aug_data)
                aug_label.append(tmp_label)
            aug_data = np.concatenate(aug_data)
            aug_label = np.concatenate(aug_label)
            aug_shuffle = np.random.permutation(len(aug_data))
            aug_data = aug_data[aug_shuffle, :, :]
            aug_label = aug_label[aug_shuffle]

            return aug_data, aug_label

        source_data = []
        source_label = []
        # to get the data of source subject
        for sub_index in range(9):
            sub_index += 1
            if sub_index != self.nSub:
                tmp = scipy.io.loadmat(self.root + 'A0%dT.mat' % sub_index)
                tmp_one_sub_data = tmp['data']
                tmp_one_sub_label = tmp['label']

                # tmp_one_sub_data = tmp_one_sub_data[250:1000, :, :]
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


        # correspond to the number of source data
        tmp_d = self.target_data
        tmp_l = self.target_label

        self.full_data = np.concatenate([tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d, tmp_d])
        self.full_label = np.concatenate([tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l, tmp_l])


        cov_all = []
        for ad_index in range(self.full_data.shape[0]):
            tmp_ad = self.full_data[ad_index, 0, :, :]
            oneone = np.dot(tmp_ad, tmp_ad.transpose())
            one_cov = oneone / np.trace(oneone)
            cov_all.append(one_cov)
        cov = np.mean(cov_all, axis=0)

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

        self.source_id = np.zeros(self.source_label.shape)
        self.source_id[:] = 1
        self.target_id = np.zeros(self.full_label.shape)
        return self.source_data, self.source_label, self.full_data, self.full_label, cov, self.test_data, self.test_label, self.source_id, self.target_id

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

        sour_img, sour_label, img, label, cov, test_data, test_label, sour_id, targ_id = self.get_data()
        # shuffle one more time
        sour_shuflle_num = np.random.permutation(len(sour_img))
        sour_img = sour_img[sour_shuflle_num, :, :, :]
        sour_label = sour_label[sour_shuflle_num]
        shuffle_num = np.random.permutation(len(img))
        img = img[shuffle_num, :, :, :]  # img is the target data
        label = label[shuffle_num]

        sour_img = torch.from_numpy(sour_img)
        sour_label = torch.from_numpy(sour_label - 1)
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        sour_id = torch.from_numpy(sour_id)
        targ_id = torch.from_numpy(targ_id)

        dataset = torch.utils.data.TensorDataset(img, label, sour_img, sour_label, sour_id, targ_id)
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

        # Train the cnn model
        for e in range(self.n_epochs):
            tacc = 0
            tnum = 0
            for i, (img, label, sour_img, sour_label, sour_id, targ_id) in enumerate(self.dataloader):

                img = Variable(img.type(self.Tensor))
                label = Variable(label.type(self.LongTensor))
                sour_img = Variable(sour_img.type(self.Tensor))
                sour_label = Variable(sour_label.type(self.LongTensor))

                sour_id = Variable(sour_id.type(self.LongTensor))
                targ_id = Variable(targ_id.type(self.LongTensor))

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

                    # Total loss
                    loss_D = loss_D_GAN

                    loss_D.backward()
                    self.optimizer_dis.step()

                # --------------
                #  Train the united networks, including the encoder and the classifier
                # --------------
                if (i + 1) % 1 == 0:
                    self.optimizer.zero_grad()
                    # encoder
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

                    loss_cls_sour = self.criterion_cls(sour_out_cls, sour_label)

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
                    # cen_feature_st = feature
                    # cen_label_st = label
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

                    loss_U = loss_Joint_cls + loss_Joint_adv + self.lambda_cen/5 * loss_Cen  

                    loss_U.backward()
                    self.optimizer.step()

                    # update centers
                    deltacA = self.update_centers(cen_feature, cen_label.cpu().numpy())

                    with torch.no_grad():
                        for ke in deltacA.keys():
                            self.centers[ke] = self.centers[ke] - self.alpha * deltacA[ke]



                torch.save(self.Feature_Extractor.state_dict(), "model/sub%d_Enc_65.pth" % self.nSub)
                torch.save(self.Classifier.state_dict(), "model/sub%d_Cls_65.pth" % self.nSub)
                torch.save(self.Discriminator.state_dict(), "model/sub%d_Dis_65.pth" % self.nSub)

            tacc = 1.0 * tacc / tnum
            gamma = 1 / 2**(int((tacc - 0.3)/0.1))

            if (e + 1) % 1 == 0:
                acc = self.test(e)
                num += 1
                averAcc += acc
                if acc > bestAcc:
                    bestAcc = acc

            print(
                'Epoch: %d     Train_targ_Accuracy: %f     Cen_loss: %.6f     Test Accuracy: %f'
                % (e, tacc, loss_Cen, acc))

        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc
        # writer.close()

    def test(self, e):
        feature_extractor = Feature_Extractor()
        classifier = Classifier()

        feature_extractor = nn.DataParallel(feature_extractor)
        classifier = nn.DataParallel(classifier)
        feature_extractor = feature_extractor.cuda()
        classifier = classifier.cuda()

        feature_extractor.load_state_dict(torch.load("model/sub%d_Enc_65.pth" % self.nSub))
        classifier.load_state_dict(torch.load("model/sub%d_Cls_65.pth" % self.nSub))

        acc = 0
        num = 0
        for i, (test_data, test_label) in enumerate(self.test_dataloader):
            test_data = Variable(test_data.type(self.Tensor))
            test_label = Variable(test_label.type(self.LongTensor))

            (feature, feature) = feature_extractor((test_data, test_data))
            Cls = classifier(feature)

            y_pred = torch.max(Cls, 1)[1]

            for k in range(len(test_label)):
                num = num + 1
                if y_pred[k] == test_label[k]:
                    acc = acc + 1
        acc = 1.0 * acc / num
        self.log_write.write(str(e) + "    " + str(acc) + "\n")

        return acc


def main():
    best = 0
    aver = 0
    result_write = open("./results/test/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()
        # i = 5
        # i = 1
        # i = 4
        datrans = DATrans(i + 1)
        bestAcc, averAcc = datrans.train()
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        best = best + bestAcc
        aver = aver + averAcc
        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))

    best = best / 9
    aver = aver / 9
    result_write.write('The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()

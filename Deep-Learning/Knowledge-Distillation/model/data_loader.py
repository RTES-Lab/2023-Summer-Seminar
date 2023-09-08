"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def fetch_dataloader(types, params):
    """
    types: 'train' 또는 'dev' 문자열을 받아, 훈련 데이터셋 또는 검증 데이터셋을 로딩한다.
    params: 하이퍼파라미터 정보를 담고 있는 객체. 배치 크기, 데이터 증강 여부, CPU/GPU 사용 여부 등을 포함한다.
    """

    
    if params.augmentation == "yes":  # RandomCrop(랜덤 크롭)과 RandomHorizontalFlip(수평 뒤집기)를 적용한다
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) #CIFAR-10 데이터셋의 RGB 채널에 대한 평균과 표준편차를 사용하여 정규화

    else:  # 데이터 증강을 적용하지 않는다
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set(검증 데이터)
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)
    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    # types 인자에 따라 훈련 또는 검증 DataLoader 반환
    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)

    devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl
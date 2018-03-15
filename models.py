# -*- coding: utf-8 -*-



# Imports.
import pickle                               as pkl
import ipdb
import numpy                                as np
import os
import sys
import torch                                as T
import torch.autograd                       as TA
import torch.cuda                           as TC
import torch.nn                             as TN
import torch.optim                          as TO
import torch.utils                          as TU
import torch.utils.data                     as TUD

from   functional                           import *
from   layers                               import *




#
# Real-valued model
#

class RealModel(TN.Module):
	def __init__(self, a):
		super(RealModel, self).__init__()
		self.a      = a
		self.conv0  = TN.Conv2d          (  1 if self.a.dataset == "mnist" else 3,
		                                        64, (3,3), padding=1)
		self.bn0    = TN.BatchNorm2d     ( 64,  affine=False)
		self.relu0  = TN.ReLU            ()
		self.conv1  = TN.Conv2d          ( 64,  64, (3,3), padding=1)
		self.bn1    = TN.BatchNorm2d     ( 64,  affine=False)
		self.relu1  = TN.ReLU            ()
		self.conv2  = TN.Conv2d          ( 64, 128, (3,3), padding=1, stride=2)
		self.bn2    = TN.BatchNorm2d     (128,  affine=False)
		self.relu2  = TN.ReLU            ()
		self.conv3  = TN.Conv2d          (128, 128, (3,3), padding=1)
		self.bn3    = TN.BatchNorm2d     (128,  affine=False)
		self.relu3  = TN.ReLU            ()
		self.conv4  = TN.Conv2d          (128, 128, (3,3), padding=1)
		self.bn4    = TN.BatchNorm2d     (128,  affine=False)
		self.relu4  = TN.ReLU            ()
		self.conv5  = TN.Conv2d          (128, 256, (3,3), padding=1, stride=2)
		self.bn5    = TN.BatchNorm2d     (256,  affine=False)
		self.relu5  = TN.ReLU            ()
		self.conv6  = TN.Conv2d          (256, 256, (3,3), padding=1)
		self.bn6    = TN.BatchNorm2d     (256,  affine=False)
		self.relu6  = TN.ReLU            ()
		self.conv7  = TN.Conv2d          (256, 256, (3,3), padding=1)
		self.bn7    = TN.BatchNorm2d     (256,  affine=False)
		self.relu7  = TN.ReLU            ()
		self.conv8  = TN.Conv2d          (256, 256, (3,3), padding=1)
		self.bn8    = TN.BatchNorm2d     (256,  affine=False)
		self.relu8  = TN.ReLU            ()
		self.conv9  = TN.Conv2d          (256,  100 if self.a.dataset == "cifar100" else 10,
		                                            (1,1), padding=0)
		self.pool   = TN.AvgPool2d       ((7,7) if self.a.dataset == "mnist" else (8,8))
		self.celoss = TN.CrossEntropyLoss()
	def forward(self, X, Y):
		v, y     = X.view(-1, 1, 28, 28), Y
		v        = self.relu0(self.bn0(self.conv0(v)))
		v        = self.relu1(self.bn1(self.conv1(v)))
		v        = self.relu2(self.bn2(self.conv2(v)))
		v        = self.relu3(self.bn3(self.conv3(v)))
		v        = self.relu4(self.bn4(self.conv4(v)))
		v        = self.relu5(self.bn5(self.conv5(v)))
		v        = self.relu6(self.bn6(self.conv6(v)))
		v        = self.relu7(self.bn7(self.conv7(v)))
		v        = self.relu8(self.bn8(self.conv8(v)))
		v        = self.pool (self.conv9(v))
		v        = v.view(-1, 10)
		
		ceLoss   = self.celoss(v, y)
		yPred    = T.max(v, 1)[1]
		batchErr = yPred.eq(y).long().sum(0)
		
		return {
			"user/ceLoss":   ceLoss,
			"user/batchErr": batchErr,
			"user/yPred":    yPred,
		}

#
# Matthieu Courbariaux model reproduction attempt
#

class MatthieuBNN(TN.Module):
	def __init__(self, a):
		super(MatthieuBNN, self).__init__()
		T.backends.cudnn.benchmark=True # Hack to hopefully make things faster
		self.a = a
		
		epsilon = 1e-4   # Some epsilon
		alpha   = 1-0.9  # Exponential moving average factor for BN.
		
		#
		# Model Layers
		#
		
		self.conv1     = Conv2dBNN     (   3,  128, (3,3), padding=1, H=1, W_LR_scale="Glorot")
		self.bn1       = TN.BatchNorm2d( 128, epsilon, alpha)
		self.tanh1     = BNNTanh       ()
		self.conv2     = Conv2dBNN     ( 128,  128, (3,3), padding=1, H=1, W_LR_scale="Glorot")
		self.maxpool2  = TN.MaxPool2d  ((2,2), stride=(2,2))
		self.bn2       = TN.BatchNorm2d( 128, epsilon, alpha)
		self.tanh2     = BNNTanh       ()
		
		self.conv3     = Conv2dBNN     ( 128,  256, (3,3), padding=1, H=1, W_LR_scale="Glorot")
		self.bn3       = TN.BatchNorm2d( 256, epsilon, alpha)
		self.tanh3     = BNNTanh       ()
		self.conv4     = Conv2dBNN     ( 256,  256, (3,3), padding=1, H=1, W_LR_scale="Glorot")
		self.maxpool4  = TN.MaxPool2d  ((2,2), stride=(2,2))
		self.bn4       = TN.BatchNorm2d( 256, epsilon, alpha)
		self.tanh4     = BNNTanh       ()
		
		self.conv5     = Conv2dBNN     ( 256,  512, (3,3), padding=1, H=1, W_LR_scale="Glorot")
		self.bn5       = TN.BatchNorm2d( 512, epsilon, alpha)
		self.tanh5     = BNNTanh       ()
		self.conv6     = Conv2dBNN     ( 512,  512, (3,3), padding=1, H=1, W_LR_scale="Glorot")
		self.maxpool6  = TN.MaxPool2d  ((2,2), stride=(2,2))
		self.bn6       = TN.BatchNorm2d( 512, epsilon, alpha)
		self.tanh6     = BNNTanh       ()
		
		self.linear7   = LinearBNN     (4*4*512, 1024, H=1, W_LR_scale="Glorot")
		self.bn7       = TN.BatchNorm2d(1024, epsilon, alpha)
		self.tanh7     = BNNTanh       ()
		self.linear8   = LinearBNN     (1024, 1024, H=1, W_LR_scale="Glorot")
		self.bn8       = TN.BatchNorm2d(1024, epsilon, alpha)
		self.tanh8     = BNNTanh       ()
		self.linear9   = LinearBNN     (1024,   10, H=1, W_LR_scale="Glorot")
		self.bn9       = TN.BatchNorm2d(  10, epsilon, alpha)
	
	
	def forward(self, X):
		self.conv1  .reconstrain()
		self.conv2  .reconstrain()
		self.conv3  .reconstrain()
		self.conv4  .reconstrain()
		self.conv5  .reconstrain()
		self.conv6  .reconstrain()
		self.linear7.reconstrain()
		self.linear8.reconstrain()
		self.linear9.reconstrain()
		
		shape = (-1, 1, 28, 28) if self.a.dataset == "mnist" else (-1, 3, 32, 32)
		v = X.view(*shape)
		
		v = v*2-1
		
		v = self.conv1   (v)
		v = self.bn1     (v)
		v = self.tanh1   (v)
		v = self.conv2   (v)
		v = self.maxpool2(v)
		v = self.bn2     (v)
		v = self.tanh2   (v)
		
		v = self.conv3   (v)
		v = self.bn3     (v)
		v = self.tanh3   (v)
		v = self.conv4   (v)
		v = self.maxpool4(v)
		v = self.bn4     (v)
		v = self.tanh4   (v)
		
		v = self.conv5   (v)
		v = self.bn5     (v)
		v = self.tanh5   (v)
		v = self.conv6   (v)
		v = self.maxpool6(v)
		v = self.bn6     (v)
		v = self.tanh6   (v)
		
		v = v.view(-1, 4*4*512)
		
		v = self.linear7 (v)
		v = self.tanh7   (v)
		v = self.linear8 (v)
		v = self.tanh8   (v)
		v = self.linear9 (v)
		
		return v
	
	def loss(self, Ypred, Y):
		onehotY   = T.zeros_like(Ypred).scatter_(1, Y.unsqueeze(1), 1)*2 - 1
		hingeLoss = T.mean(T.clamp(1.0 - Ypred*onehotY, min=0)**2)
		return hingeLoss




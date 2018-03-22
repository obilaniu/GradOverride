# -*- coding: utf-8 -*-
import numpy            as np
import torch

from   torch.nn     import (Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, ReLU,
                            CrossEntropyLoss,)

from   functional   import *
from   layers       import *



#
# Hack to hopefully make things faster. Enable autoselection of cuDNN
# algorithms.
#
torch.backends.cudnn.benchmark=True



#
# For the sake of ease of use, all models below inherit from this class, which
# exposes a constrain() method that allows re-applying the module's constraints
# to its parameters.
#
class ModelConstrained(torch.nn.Module):
	def constrain(self):
		def fn(module):
			if module is not self and hasattr(module, "constrain"):
				module.constrain()
		
		self.apply(fn)

#
# Real-valued model
#
class ModelReal(ModelConstrained):
	def __init__(self, a):
		super().__init__()
		self.a      = a
		
		inChan      =     1 if self.a.dataset == "mnist"    else  3
		outChan     =   100 if self.a.dataset == "cifar100" else 10
		outPool     = (7,7) if self.a.dataset == "mnist"    else (8,8)
		
		self.conv0  = Conv2d          (inChan, 64, (3,3), padding=1)
		self.bn0    = BatchNorm2d     ( 64,  affine=False)
		self.relu0  = ReLU            ()
		self.conv1  = Conv2d          ( 64,  64, (3,3), padding=1)
		self.bn1    = BatchNorm2d     ( 64,  affine=False)
		self.relu1  = ReLU            ()
		self.conv2  = Conv2d          ( 64, 128, (3,3), padding=1, stride=2)
		self.bn2    = BatchNorm2d     (128,  affine=False)
		self.relu2  = ReLU            ()
		self.conv3  = Conv2d          (128, 128, (3,3), padding=1)
		self.bn3    = BatchNorm2d     (128,  affine=False)
		self.relu3  = ReLU            ()
		self.conv4  = Conv2d          (128, 128, (3,3), padding=1)
		self.bn4    = BatchNorm2d     (128,  affine=False)
		self.relu4  = ReLU            ()
		self.conv5  = Conv2d          (128, 256, (3,3), padding=1, stride=2)
		self.bn5    = BatchNorm2d     (256,  affine=False)
		self.relu5  = ReLU            ()
		self.conv6  = Conv2d          (256, 256, (3,3), padding=1)
		self.bn6    = BatchNorm2d     (256,  affine=False)
		self.relu6  = ReLU            ()
		self.conv7  = Conv2d          (256, 256, (3,3), padding=1)
		self.bn7    = BatchNorm2d     (256,  affine=False)
		self.relu7  = ReLU            ()
		self.conv8  = Conv2d          (256, 256, (3,3), padding=1)
		self.bn8    = BatchNorm2d     (256,  affine=False)
		self.relu8  = ReLU            ()
		self.conv9  = Conv2d          (256,  outChan, (1,1), padding=0)
		self.pool   = AvgPool2d       (outPool)
		self.loss   = CrossEntropyLoss()
	
	def forward(self, X):
		shape   = (-1, 1, 28, 28) if self.a.dataset == "mnist" else (-1, 3, 32, 32)
		outChan = 100 if self.a.dataset == "cifar100" else 10
		
		v = X.view(*shape)
		v = self.relu0(self.bn0(self.conv0(v)))
		v = self.relu1(self.bn1(self.conv1(v)))
		v = self.relu2(self.bn2(self.conv2(v)))
		v = self.relu3(self.bn3(self.conv3(v)))
		v = self.relu4(self.bn4(self.conv4(v)))
		v = self.relu5(self.bn5(self.conv5(v)))
		v = self.relu6(self.bn6(self.conv6(v)))
		v = self.relu7(self.bn7(self.conv7(v)))
		v = self.relu8(self.bn8(self.conv8(v)))
		v = self.pool (self.conv9(v))
		v = v.view(-1, outChan)
		return v


#
# Binary-valued model.
#
class ModelBNN(ModelConstrained):
	"""
	https://arxiv.org/pdf/1602.02830.pdf
	"""
	def __init__(self, a):
		super().__init__()
		self.a = a
		
		override       = self.a.override
		inChan         =     1 if self.a.dataset == "mnist"    else  3
		outChan        =   100 if self.a.dataset == "cifar100" else 10
		epsilon        = 1e-4   # Some epsilon
		alpha          = 1-0.9  # Exponential moving average factor for BN.
		
		self.conv1     = Conv2dBNN  (inChan, 128, (3,3), padding=1, H=1, W_LR_scale="Glorot", override=override)
		self.bn1       = BatchNorm2d( 128, epsilon, alpha)
		self.tanh1     = SignBNN    (override)
		self.conv2     = Conv2dBNN  ( 128,  128, (3,3), padding=1, H=1, W_LR_scale="Glorot", override=override)
		self.maxpool2  = MaxPool2d  ((2,2), stride=(2,2))
		self.bn2       = BatchNorm2d( 128, epsilon, alpha)
		self.tanh2     = SignBNN    (override)
		
		self.conv3     = Conv2dBNN  ( 128,  256, (3,3), padding=1, H=1, W_LR_scale="Glorot", override=override)
		self.bn3       = BatchNorm2d( 256, epsilon, alpha)
		self.tanh3     = SignBNN    (override)
		self.conv4     = Conv2dBNN  ( 256,  256, (3,3), padding=1, H=1, W_LR_scale="Glorot", override=override)
		self.maxpool4  = MaxPool2d  ((2,2), stride=(2,2))
		self.bn4       = BatchNorm2d( 256, epsilon, alpha)
		self.tanh4     = SignBNN    (override)
		
		self.conv5     = Conv2dBNN  ( 256,  512, (3,3), padding=1, H=1, W_LR_scale="Glorot", override=override)
		self.bn5       = BatchNorm2d( 512, epsilon, alpha)
		self.tanh5     = SignBNN    (override)
		self.conv6     = Conv2dBNN  ( 512,  512, (3,3), padding=1, H=1, W_LR_scale="Glorot", override=override)
		self.maxpool6  = MaxPool2d  ((2,2), stride=(2,2))
		self.bn6       = BatchNorm2d( 512, epsilon, alpha)
		self.tanh6     = SignBNN    (override)
		
		self.linear7   = LinearBNN  (4*4*512, 1024, H=1, W_LR_scale="Glorot", override=override)
		self.tanh7     = SignBNN    (override)
		self.linear8   = LinearBNN  (1024, 1024, H=1, W_LR_scale="Glorot", override=override)
		self.tanh8     = SignBNN    (override)
		self.linear9   = LinearBNN  (1024,  outChan, H=1, W_LR_scale="Glorot", override=override)
	
	
	def forward(self, X):
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
		
		v = v.view(v.size(0), -1)
		
		v = self.linear7 (v)
		v = self.tanh7   (v)
		v = self.linear8 (v)
		v = self.tanh8   (v)
		v = self.linear9 (v)
		
		return v
	
	def loss(self, Ypred, Y):
		onehotY   = torch.zeros_like(Ypred).scatter_(1, Y.unsqueeze(1), 1)*2 - 1
		hingeLoss = torch.mean(torch.clamp(1.0 - Ypred*onehotY, min=0)**2)
		return hingeLoss




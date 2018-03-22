# -*- coding: utf-8 -*-
import numpy                                as np
import torch
import torch.nn.functional                  as TNF

from   functional                       import *



#
# PyTorch Convolution Layers
#

class Conv2dBNN(torch.nn.Conv2d):
	"""
	Convolution layer for BinaryNet.
	"""
	
	def __init__(self, in_channels,
	                   out_channels,
	                   kernel_size,
	                   stride       = 1,
	                   padding      = 0,
	                   dilation     = 1,
	                   groups       = 1,
	                   bias         = True,
	                   H            = 1.0,
	                   W_LR_scale   = "Glorot",
	                   override     = "matt"):
		#
		# Fan-in/fan-out computation
		#
		num_inputs = in_channels
		num_units  = out_channels
		for x in kernel_size:
			num_inputs *= x
			num_units  *= x
		
		if H == "Glorot":
			self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
		else:
			self.H          = H
		
		if W_LR_scale == "Glorot":
			self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
		else:
			self.W_LR_scale = self.H
		
		self.override = override
		
		super().__init__(in_channels, out_channels, kernel_size,
		                 stride, padding, dilation, groups, bias)
		self.reset_parameters()
	
	def reset_parameters(self):
		self.weight.data.uniform_(-self.H, +self.H)
		if isinstance(self.bias, torch.nn.Parameter):
			self.bias.data.zero_()
	
	def constrain(self):
		self.weight.data.clamp_(-self.H, +self.H)
	
	def forward(self, x):
		if   self.override == "matt":
			Wb = bnn_sign(self.weight/self.H)*self.H
		elif self.override == "pass":
			Wb = bnn_sign_pass(self.weight/self.H)*self.H
		return TNF.conv2d(x, Wb, self.bias, self.stride, self.padding,
		                  self.dilation, self.groups)



#
# PyTorch Dense Layers
#

class LinearBNN(torch.nn.Linear):
	"""
	Linear/Dense layer for BinaryNet.
	"""
	
	def __init__(self, in_channels,
	                   out_channels,
	                   bias         = True,
	                   H            = 1.0,
	                   W_LR_scale   = "Glorot",
	                   override     = "matt"):
		#
		# Fan-in/fan-out computation
		#
		num_inputs = in_channels
		num_units  = out_channels
		
		if H == "Glorot":
			self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
		else:
			self.H          = H
		
		if W_LR_scale == "Glorot":
			self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
		else:
			self.W_LR_scale = self.H
		
		self.override = override
		
		super().__init__(in_channels, out_channels, bias)
		self.reset_parameters()
	
	def reset_parameters(self):
		self.weight.data.uniform_(-self.H, +self.H)
		if isinstance(self.bias, torch.nn.Parameter):
			self.bias.data.zero_()
	
	def constrain(self):
		self.weight.data.clamp_(-self.H, +self.H)
	
	def forward(self, input):
		if   self.override == "matt":
			Wb = bnn_sign(self.weight/self.H)*self.H
		elif self.override == "pass":
			Wb = bnn_sign_pass(self.weight/self.H)*self.H
		return TNF.linear(input, Wb, self.bias)



#
# PyTorch Non-Linearities
#

class SignBNN(torch.nn.Module):
	def __init__(self, override="matt"):
		super().__init__()
		self.override = override
	def forward(self, x):
		if   self.override == "matt":
			return bnn_sign(x)
		elif self.override == "pass":
			return bnn_sign_pass(x)



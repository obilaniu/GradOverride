# -*- coding: utf-8 -*-
import torch


class BNNSign(torch.autograd.Function):
	"""
	BinaryNet q = Sign(r) with gradient override.
	Equation (1) and (4) of https://arxiv.org/pdf/1602.02830.pdf
	"""
	
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		return x.sign()
	
	@staticmethod
	def backward(ctx, dx):
		x, = ctx.saved_variables
		
		gt1  = x > +1
		lsm1 = x < -1
		gi   = 1-gt1.float()-lsm1.float()
		
		return gi*dx

bnn_sign = BNNSign.apply


class BNNSignPass(torch.autograd.Function):
	"""
	BinaryNet q = Sign(r) with gradient override.
	Same as BNNSign except that gradient is passed through unchanged.
	"""
	
	@staticmethod
	def forward(ctx, x):
		return x.sign()
	
	@staticmethod
	def backward(ctx, dx):
		return dx

bnn_sign_pass = BNNSignPass.apply


class PACTFunction(torch.autograd.Function):
	"""
	Parametrized Clipping Activation Function
	https://arxiv.org/pdf/1805.06085.pdf
	"""
	
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.save_for_backward(x, alpha)
		return x.clamp(min=0.0).min(alpha)
	
	@staticmethod
	def backward(ctx, dLdy):
		x, alpha = ctx.saved_variables
		
		lt0      = x < 0
		gta      = x > alpha
		gi       = 1.0-lt0.float()-gta.float()
		
		dLdx     = dLdy*gi
		dLdalpha = torch.sum(dLdy*x.ge(alpha).float())
		return dLdx, dLdalpha

pact          = PACTFunction.apply

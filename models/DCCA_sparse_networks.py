import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt

def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', gain=0.02):
	net = net
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			elif init_type == 'pretrained':
				pass
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)

	for root_child in net.children():
		for children in root_child.children():
			if children in root_child.need_initialization:
				init_weights(children, init_type, gain=init_gain)
	return net

def define_DCCASparseNet(rgb_enc=True, depth_enc=True, depth_dec=True, norm='batch', init_type='xavier', init_gain=0.02, gpu_ids=[]):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)
	net = DCCASparsenetGenerator(rgb_enc=rgb_enc, depth_enc=depth_enc, depth_dec=depth_dec)
	return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class SAConv(nn.Module):
	# Convolution layer for sparse data
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
		super(SAConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
		nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
		self.pool.require_grad = False

	def forward(self, input):
		x, m = input
		x = x * m
		x = self.conv(x)
		weights = torch.ones(torch.Size([1, 1, 3, 3])).cuda()
		mc = F.conv2d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		mc = torch.clamp(mc, min=1e-5)
		mc = 1. / mc * 9

		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		m = self.pool(m)

		return x, m

class SAConvBlock(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1, bias=True):
		super(SAConvBlock, self).__init__()
		self.sparse_conv = SAConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input):
		x, m = input
		x, m = self.sparse_conv((x, m))
		assert (m.size(1)==1)
		x = self.relu(x)
		
		return x, m

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))
        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

def make_layers_from_size(sizes):
	layers = []
	for size in sizes:
		layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1), nn.BatchNorm2d(size[1],momentum = 0.1), nn.ReLU(inplace=True)]
	return nn.Sequential(*layers)

def make_blocks_from_names(names,in_dim,out_dim):
	layers = []
	if names[0] == "block1" or names[0] == "block2":
		layers += [SAConvBlock(in_dim, out_dim, 3,stride = 1)]
		layers += [SAConvBlock(out_dim, out_dim, 3,stride = 1)]
	else:
		layers += [SAConvBlock(in_dim, out_dim, 3,stride = 1)]
		layers += [SAConvBlock(out_dim, out_dim, 3,stride = 1)]
		layers += [SAConvBlock(out_dim, out_dim, 3,stride = 1)]
	return nn.Sequential(*layers)

class DCCASparsenetGenerator(nn.Module):
	def __init__(self, rgb_enc=True, depth_enc=True, depth_dec=True):
		super(DCCASparsenetGenerator, self).__init__()
		#batchNorm_momentum = 0.1
		self.need_initialization = []

		if rgb_enc :
			##### RGB ENCODER ####
			self.CBR1_RGB_ENC = make_blocks_from_names(["block1"], 3,64)
			self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

			self.CBR2_RGB_ENC = make_blocks_from_names(["block2"], 64, 128)
			self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

			self.CBR3_RGB_ENC = make_blocks_from_names(["block3"], 128, 256)
			self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
			self.dropout3 = nn.Dropout(p=0.4)

			self.CBR4_RGB_ENC = make_blocks_from_names(["block4"], 256, 512)
			self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
			self.dropout4 = nn.Dropout(p=0.4)

			self.CBR5_RGB_ENC = make_blocks_from_names(["block5"], 512, 512)
			self.dropout5 = nn.Dropout(p=0.4)

			self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		if depth_enc :

			self.CBR1_DEPTH_ENC = make_blocks_from_names(["block1"], 1, 64)
			self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
 
			self.CBR2_DEPTH_ENC = make_blocks_from_names(["block2"], 64, 128)
			self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

			self.CBR3_DEPTH_ENC = make_blocks_from_names(["block3"], 128, 256)
			self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

			self.CBR4_DEPTH_ENC = make_blocks_from_names(["block4"], 256, 512)
			self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

			self.CBR5_DEPTH_ENC = make_blocks_from_names(["block5"], 512, 512)

			self.pool5_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		if  depth_dec :
			####  DECODER  ####
			self.Transform = make_blocks_from_names(["block1"],512, 512)
			self.decoder = DeConv(1024, 3)
			self.conv3 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=False)
			self.bilinear = nn.Upsample((228,912), mode='bilinear', align_corners=True)

		self.need_initialization.append(self.decoder)
		self.need_initialization.append(self.conv3)

	def forward(self, sparse_rgb,sparse_d,mask,rgb,d):

		########  DEPTH ENCODER  ########
		x_1,m_d = self.CBR1_DEPTH_ENC((sparse_d,mask))
		x, id1_d = self.pool1_d(x_1)
		m_d,_ = self.pool1_d(m_d )

		x_2,m_d  = self.CBR2_DEPTH_ENC((x,m_d ))
		x, id2_d = self.pool2_d(x_2)
		m_d,_  = self.pool2_d(m_d )
		
		x_3,m_d  = self.CBR3_DEPTH_ENC((x,m_d ))
		x, id3_d = self.pool4_d(x_3)
		m_d,_  = self.pool3_d(m_d )

		x_4,m_d = self.CBR4_DEPTH_ENC((x,m_d ))
		x, id4_d = self.pool4_d(x_4)
		m_d,_  = self.pool4_d(m_d )

		x_5,m_d  = self.CBR5_DEPTH_ENC((x,m_d ))
		x_dataview, id5_d = self.pool5_d(x_5)
		m_d,_  = self.pool5_d(m_d )

		########  RGB ENCODER  ########
		y_1,m_r = self.CBR1_RGB_ENC((sparse_rgb,mask))
		y, id1 = self.pool1(y_1)
		m_r,_ = self.pool1(m_r)

		y_2,m_r = self.CBR2_RGB_ENC((y,m_r))
		y, id2 = self.pool2(y_2)
		m_r,_ = self.pool2(m_r)

		y_3,m_r = self.CBR3_RGB_ENC((y,m_r))
		y, id3 = self.pool3(y_3)
		m_r,_ = self.pool3(m_r)

		y_4,m_r = self.CBR4_RGB_ENC((y,m_r))
		y, id4 = self.pool4(y_4)
		m_r,_ = self.pool4(m_r)

		y_5,m_r = self.CBR5_RGB_ENC((y,m_r))
		y_dataview, id5 = self.pool5(y_5)
		m_r,_ = self.pool5(m_r)

		########  MISSING DATA ENCODER  ########
		inverse_mask = torch.ones_like(mask)-mask
		inverse_rgb = rgb*inverse_mask

		ym_1,m_m = self.CBR1_RGB_ENC((inverse_rgb,inverse_mask))
		ym, id1_m = self.pool1(ym_1)
		m_m,_ = self.pool1(m_m)

		ym_2,m_m  = self.CBR2_RGB_ENC((ym,m_m ))
		ym, id2_m = self.pool2(ym_2)
		m_m,_  = self.pool2(m_m)

		ym_3,m_m  = self.CBR3_RGB_ENC((ym,m_m ))
		ym, id3_m = self.pool4(ym_3)
		m_m,_  = self.pool3(m_m)
		
		ym_4,m_m = self.CBR4_RGB_ENC((ym,m_m ))
		ym, id4_m = self.pool4(ym_4)
		m_m,_  = self.pool4(m_m)
		
		ym_5,m_m  = self.CBR5_RGB_ENC((ym,m_m ))
		ym_dataview, id5_m = self.pool5(ym_5)
		m_m,_  = self.pool5(m_m)
		
		########  Transformer  ########
		x_trans, m_trans  = self.Transform((y_dataview,m_r))
		xm_trans, mm_trans  = self.Transform((ym_dataview,m_r))

		########  DECODER  ########
		x = self.decoder(torch.cat((x_dataview,xm_trans),1))
		x = self.conv3(x)
		depth_est = self.bilinear(x)
		
		return x_dataview, y_dataview, x_trans, depth_est

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class TransformLoss(nn.Module):
    def __init__(self):
        super(TransformLoss, self).__init__()

    def forward(self, f_in, f_target):
        assert f_in.dim() == f_target.dim(), "inconsistent dimensions"
        diff = f_in - f_target
        self.loss = (diff ** 2).mean()
        return self.loss

class SmoothLoss(nn.Module):
	def __init__(self):
		super(SmoothLoss, self).__init__()

	def forward(self, pred_map):
	    def gradient(pred):
	        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
	        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
	        return D_dx, D_dy

	    if type(pred_map) not in [tuple, list]:
	        pred_map = [pred_map]

	    loss = 0
	    weight = 1.

	    for scaled_map in pred_map:
	        dx, dy = gradient(scaled_map)
	        dx2, dxdy = gradient(dx)
	        dydx, dy2 = gradient(dy)
	        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
	        weight /= 2.3  # don't ask me why it works better
	    return loss

class DCCA_2D_Loss(nn.Module):
	def __init__(self,outdim_size, use_all_singular_values, device):
		super(DCCA_2D_Loss, self).__init__()
		self.outdim_size = outdim_size
		self.use_all_singular_values = use_all_singular_values
		self.device = device

	def __call__(self, data_view1, data_view2):
		H1 = data_view1.view(data_view1.size(0)*data_view1.size(1),data_view1.size(2),data_view1.size(3))
		H2 = data_view2.view(data_view2.size(0)*data_view2.size(1),data_view2.size(2),data_view2.size(3))

		r1 = 1e-4
		r2 = 1e-4
		eps = 1e-12
		corr_sum = 0
		o1 = o2 = H1.size(1)

		m = H1.size(0)
		n = H1.size(1)
		
		H1bar = H1 - (1.0 / m) * H1
		H2bar = H2 - (1.0 / m) * H2
		Hat12 = torch.zeros(m,n,n).cuda()
		Hat11 = torch.zeros(m,n,n).cuda()
		Hat22 = torch.zeros(m,n,n).cuda()

		for i in range(m):
			Hat11[i] = torch.matmul(H1bar[i],H1bar.transpose(1,2)[i])
			Hat12[i] = torch.matmul(H1bar[i],H2bar.transpose(1,2)[i])
			Hat22[i] = torch.matmul(H2bar[i],H2bar.transpose(1,2)[i])

		SigmaHat12 = (1.0 / (m - 1)) * torch.mean(Hat12,dim=0)
		SigmaHat11 = (1.0 / (m - 1)) * torch.mean(Hat11,dim=0)+ r1 * torch.eye(o1, device=self.device)
		SigmaHat22 = (1.0 / (m - 1)) * torch.mean(Hat22,dim=0) + r2 * torch.eye(o2, device=self.device)

		# Calculating the root inverse of covariance matrices by using eigen decomposition
		[D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
		[D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

		# Added to increase stability
		posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
		D1 = D1[posInd1]
		V1 = V1[:, posInd1]
		posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
		D2 = D2[posInd2]
		V2 = V2[:, posInd2]
		SigmaHat11RootInv = torch.matmul(
			torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
		SigmaHat22RootInv = torch.matmul(
			torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

		Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
								   SigmaHat12), SigmaHat22RootInv)

		if self.use_all_singular_values:
			# all singular values are used to calculate the correlation
			corr = torch.sqrt(torch.trace(torch.matmul(Tval.t(), Tval)))
		else:
			# just the top self.outdim_size singular values are used
			U, V = torch.symeig(torch.matmul(Tval.t(), Tval), eigenvectors=True)
			U = U[torch.gt(U, eps).nonzero()[:, 0]]
			U = U.topk(self.outdim_size)[0]
			corr = torch.sum(torch.sqrt(U))
		return -corr
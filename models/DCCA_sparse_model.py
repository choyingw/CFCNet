import torch
from .base_model import BaseModel
from . import DCCA_sparse_networks
import numpy as np
import os
import math

class DCCASparseModel(BaseModel):
	def name(self):
		return 'DCCASparseNetModel'

	@staticmethod
	def modify_commandline_options(parser, is_train=True):

		# changing the default values
		if is_train:
			parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
		return parser

	def initialize(self, opt, dataset):
		BaseModel.initialize(self, opt)
		
		self.x_dataview = None
		self.y_dataview = None
		self.depth_est = None
		self.loss_dcca = 0
		self.loss_l1 = 0
		self.loss_mse = None
		self.loss_smooth = None
		self.result = None
		self.test_result = None
		self.average = None
		self.test_average = None

		self.isTrain = opt.isTrain
		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['mse','dcca','total','transform','smooth']
		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		self.visual_names = ['rgb_image','depth_image','mask','output']
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		self.model_names = ['DCCASparseNet']

		# load/define networks
		self.netDCCASparseNet = DCCA_sparse_networks.define_DCCASparseNet(rgb_enc=True, depth_enc=True, depth_dec=True, norm=opt.norm, init_type=opt.init_type,	init_gain= opt.init_gain, gpu_ids= self.gpu_ids)
        # define loss functions
		self.criterionDCCA = DCCA_sparse_networks.DCCA_2D_Loss(outdim_size = 60,use_all_singular_values = True, device=self.device).to(self.device)
		self.MSE = DCCA_sparse_networks.MaskedMSELoss()
		self.SMOOTH = DCCA_sparse_networks.SmoothLoss()
		self.TransformLoss = DCCA_sparse_networks.TransformLoss()

		if self.isTrain:
			# initialize optimizers
			self.optimizers = []
			self.optimizer_DCCASparseNet = torch.optim.SGD(self.netDCCASparseNet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
			self.optimizers.append(self.optimizer_DCCASparseNet)

	def set_input(self, input):
		self.rgb_image = input['rgb_image'].to(self.device)
		self.depth_image = input['depth_image'].to(self.device)
		self.mask = input['mask'].to(self.device)
		self.image_paths = input['path']

	def set_new_input(self, input,target):
		self.rgb_image = input[:,:3,:,:].to(self.device)
		self.sparse_rgb = input[:,4:7,:,:].to(self.device)
		self.depth_image = target.to(self.device)
		self.sparse_depth = input[:,3,:,:].to(self.device).unsqueeze(1)
		self.mask = input[:,7,:,:].to(self.device).unsqueeze(1)

	def forward(self):
		self.x_dataview,self.y_dataview,self.x_trans,self.depth_est= self.netDCCASparseNet(self.sparse_rgb,self.sparse_depth,self.mask,self.rgb_image,self.depth_image)
		
	def get_loss(self):
		self.loss_dcca = self.criterionDCCA(self.x_dataview,self.y_dataview)
		self.loss_mse = self.MSE(self.depth_est,self.depth_image)
		self.loss_smooth = self.SMOOTH(self.depth_est)
		self.loss_transform = self.TransformLoss(self.x_trans, self.x_dataview)
		self.loss_total = self.loss_mse + self.loss_dcca + self.loss_transform + 0.1*self.loss_smooth 

	def backward(self):
		self.loss_total.backward()

	def pure_backward(self):
		self.loss_dcca.backward()

	def init_test_eval(self):
		self.test_result = Result()
		self.test_average = AverageMeter()

	def init_eval(self):
		self.result = Result()
		self.average = AverageMeter()
	
	def depth_evaluation(self):
		self.result.evaluate(self.depth_est.data, self.depth_image.data)
		self.average.update(self.result, self.sparse_rgb.size(0))

	def test_depth_evaluation(self):
		self.test_result.evaluate(self.depth_est.data, self.depth_image.data)
		self.test_average.update(self.test_result, self.sparse_rgb.size(0))
		print()

	def print_test_depth_evaluation(self):
		message = 'RMSE={result.rmse:.4f}({average.rmse:.4f}) \
MAE={result.mae:.4f}({average.mae:.4f}) \
Delta1={result.delta1:.4f}({average.delta1:.4f}) \
REL={result.absrel:.4f}({average.absrel:.4f}) \
Lg10={result.lg10:.4f}({average.lg10:.4f})'.format(result=self.test_result, average=self.test_average.average())
		print(message)
		return message

	def print_depth_evaluation(self):
		message = 'RMSE={result.rmse:.4f}({average.rmse:.4f}) \
MAE={result.mae:.4f}({average.mae:.4f}) \
Delta1={result.delta1:.4f}({average.delta1:.4f}) \
REL={result.absrel:.4f}({average.absrel:.4f}) \
Lg10={result.lg10:.4f}({average.lg10:.4f})'.format(result=self.result, average=self.average.average())
		print(message)
		return message

	def optimize_parameters(self):
		self.forward()
		self.depth_evaluation()
		self.set_requires_grad(self.netDCCASparseNet, True)
		self.get_loss()
		self.optimizer_DCCASparseNet.zero_grad()
		# update DCCAnet
		self.backward()
		self.optimizer_DCCASparseNet.step()


####### Metrics ########
def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target>0 
        output = output[valid_mask]
        target = target[valid_mask]

        new_output = output[target<=50]
        new_target = target[target<=50]  
        target = new_target
        output = new_output

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, n=1):
        self.count += n
        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg

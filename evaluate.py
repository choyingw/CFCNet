#!/usr/bin/env python
import time
from options.options import AdvanceOptions
from models import create_model
from util.visualizer import Visualizer
from dataloaders.nyu_dataloader import NYUDataset
from dataloaders.kitti_dataloader import KITTIDataset
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import numpy as np
import random
import torch
import cv2
import utils
import os

# def colored_depthmap(depth, d_min=None, d_max=None):
#    if d_min is None:
# 	   d_min = np.min(depth)
#    if d_max is None:
# 	   d_max = np.max(depth)
#    depth_relative = (depth - d_min) / (d_max - d_min)
#    return 255 * plt.cm.viridis(depth_relative)[:,:,:3] # H, W, C

# def merge_into_row_with_pred_visualize(input, depth_input, rgb_sparse,depth_target, depth_est):
#    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
#    rgb_sparse = 255 * np.transpose(np.squeeze(rgb_sparse.cpu().numpy()), (1,2,0))
#    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
#    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
#    depth_pred_cpu = np.squeeze(depth_est.cpu().numpy())

#    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
#    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.min(depth_pred_cpu))
#    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
#    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
#    depth_pred_col = colored_depthmap(depth_target_cpu, d_min, d_max)

#    img_merge = np.hstack([rgb, rgb_sparse,depth_input_col, depth_target_col,depth_pred_col])

#    return img_merge

if __name__ == '__main__':
	test_opt = AdvanceOptions().parse(False)

	sparsifier = UniformSampling(test_opt.nP, max_depth=np.inf)
	#sparsifier = SimulatedStereo(100, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1)
	test_dataset = KITTIDataset(test_opt.test_path, type='val',
			modality='rgbdm', sparsifier=sparsifier)

	### Please use this dataloder if you want to use NYU
	# test_dataset = NYUDataset(test_opt.test_path, type='val',
	# 		modality='rgbdm', sparsifier=sparsifier)


	test_opt.phase = 'val'
	test_opt.batch_size = 1
	test_opt.num_threads = 1
	test_opt.serial_batches = True
	test_opt.no_flip = True

	test_data_loader = torch.utils.data.DataLoader(test_dataset,
		batch_size=test_opt.batch_size, shuffle=False, num_workers=test_opt.num_threads, pin_memory=True)

	test_dataset_size = len(test_data_loader)
	print('#test images = %d' % test_dataset_size)

	model = create_model(test_opt, test_dataset)
	model.eval()
	model.setup(test_opt)
	visualizer = Visualizer(test_opt)
	test_loss_iter = []
	gts = None
	preds = None
	epoch_iter = 0
	model.init_test_eval()
	epoch = 0
	num = 5 # How many images to save in an image
	if not os.path.exists('vis'):
		os.makedirs('vis')
	with torch.no_grad():
		iterator = iter(test_data_loader)
		i = 0
		while True:
			try:  # Some images couldn't sample more than defined nP points under Stereo sampling
				next_batch = next(iterator)
			except IndexError:
				print("Catch and Skip!")
				continue
			except StopIteration:
				break

			data, target = next_batch[0], next_batch[1]
			model.set_new_input(data,target)
			model.forward()
			model.test_depth_evaluation()
			model.get_loss()
			epoch_iter += test_opt.batch_size
			losses = model.get_current_losses()
			test_loss_iter.append(model.loss_dcca.item())

			rgb_input = model.rgb_image
			depth_input = model.sparse_depth
			rgb_sparse = model.sparse_rgb
			depth_target = model.depth_image
			depth_est = model.depth_est

			### These part save image in vis/ folder
			if i%num == 0:
				img_merge = utils.merge_into_row_with_pred_visualize(rgb_input, depth_input, rgb_sparse,depth_target, depth_est)
			elif i%num < num-1:
				row = utils.merge_into_row_with_pred_visualize(rgb_input, depth_input, rgb_sparse,depth_target, depth_est)
				img_merge = utils.add_row(img_merge, row)
			elif i%num == num-1:
				filename = 'vis/'+str(i)+'.png'
				utils.save_image(img_merge, filename)

			i += 1

			print('test epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(test_dataset) * test_opt.batch_size), end='\r')
			print(
		  'RMSE={result.rmse:.4f}({average.rmse:.4f}) '
		  'MSE={result.mse:.4f}({average.mse:.4f}) '
		  'MAE={result.mae:.4f}({average.mae:.4f}) '
		  'Delta1={result.delta1:.4f}({average.delta1:.4f}) '
		  'Delta2={result.delta2:.4f}({average.delta2:.4f}) '
		  'Delta3={result.delta3:.4f}({average.delta3:.4f}) '
		  'REL={result.absrel:.4f}({average.absrel:.4f}) '
		  'Lg10={result.lg10:.4f}({average.lg10:.4f}) '.format(
		 result=model.test_result, average=model.test_average.average()))
	avg_test_loss = np.mean(np.asarray(test_loss_iter))

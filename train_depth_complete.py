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

if __name__ == '__main__':
	train_opt = AdvanceOptions().parse(True)

	# The SimulatedStereo class is also provided to subsample to stereo points
	sparsifier = UniformSampling(train_opt.nP, max_depth=np.inf)

	train_dataset = KITTIDataset(train_opt.train_path, type='train',
                modality='rgbdm', sparsifier=sparsifier)
	test_dataset = KITTIDataset(train_opt.test_path, type='val',
            modality='rgbdm', sparsifier=sparsifier)
	## Please use this dataloder if you want to use NYU
	# train_dataset = NYUDataset(train_opt.train_path, type='train',
 #                modality='rgbdm', sparsifier=sparsifier)
	## Please use this dataloder if you want to use NYU
	# test_dataset = NYUDataset(train_opt.test_path, type='val',
 #            modality='rgbdm', sparsifier=sparsifier)

	train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_opt.batch_size, shuffle=True,
            num_workers=train_opt.num_threads, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(train_opt.seed + work_id))
	test_opt = AdvanceOptions().parse(True)
	test_opt.phase = 'val'
	test_opt.batch_size = 1
	test_opt.num_threads = 1
	test_opt.serial_batches = True
	test_opt.no_flip = True

	test_data_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=test_opt.batch_size, shuffle=False, num_workers=test_opt.num_threads, pin_memory=True)

	train_dataset_size = len(train_data_loader)
	print('#training images = %d' % train_dataset_size)
	test_dataset_size = len(test_data_loader)
	print('#test images = %d' % test_dataset_size)

	model = create_model(train_opt, train_dataset)
	model.setup(train_opt)
	visualizer = Visualizer(train_opt)
	total_steps = 0
	for epoch in range(train_opt.epoch_count, train_opt.niter + 1):
		model.train()
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0
		model.init_eval()
		iterator = iter(train_data_loader)
		while True:
			try:  # Some images couldn't sample more than defined nP points under Stereo sampling
				next_batch = next(iterator)
			except IndexError:
				print("Catch and Skip!")
				continue
			except StopIteration:
				break
			data, target = next_batch[0], next_batch[1]

			iter_start_time = time.time()
			if total_steps % train_opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			total_steps += train_opt.batch_size
			epoch_iter += train_opt.batch_size
			model.set_new_input(data,target)
			model.optimize_parameters()

			if total_steps % train_opt.print_freq == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / train_opt.batch_size
				visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
				message = model.print_depth_evaluation()
				visualizer.print_current_depth_evaluation(message)
				print()

			iter_data_time = time.time()

		print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.niter, time.time() - epoch_start_time))
		model.update_learning_rate()
		if epoch  and epoch % train_opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

			model.eval()
			test_loss_iter = []
			gts = None
			preds = None
			epoch_iter = 0
			model.init_test_eval()
			with torch.no_grad():
				iterator = iter(test_data_loader)
				while True:
					try: # Some images couldn't sample more than defined nP points under Stereo sampling
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
					print('test epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(test_dataset) * test_opt.batch_size), end='\r')
					message = model.print_test_depth_evaluation()
					visualizer.print_current_depth_evaluation(message)
					print(
                  'RMSE={result.rmse:.4f}({average.rmse:.4f}) '
                  'MAE={result.mae:.4f}({average.mae:.4f}) '
                  'Delta1={result.delta1:.4f}({average.delta1:.4f}) '
                  'REL={result.absrel:.4f}({average.absrel:.4f}) '
                  'Lg10={result.lg10:.4f}({average.lg10:.4f}) '.format(
                 result=model.test_result, average=model.test_average.average()))
			avg_test_loss = np.mean(np.asarray(test_loss_iter))

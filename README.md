# Deep RGB-D Canonical Correlation Analysis ForSparse Depth Completion
This is the code for our NeurIPS 2019 paper. [<a href="https://arxiv.org/abs/1906.08967">Arxiv</a>]

# Prerequisites
	Linux
	Python 3
	PyTorch 1.0+
	NVIDIA GPU + CUDA CuDNN 

# Getting Started

Installation:
	Clone this repo and install other dependencies by `pip install -r requirements.txt`.

Data Preparation: 
	Please refer to [<a href="http://www.cvlibs.net/datasets/kitti/index.php">KITTI</a>] or [<a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html">NYU Depth V2</a>] and process them into h5 files. <a href="https://github.com/fangchangma/sparse-to-dense.pytorch">Here</a> also provides preprocessed data.

# Train/Evaluation:

For training, please run

	python3 train_depth_complete.py --name kitti --checkpoints_dir [path to save_dir] --train_path [train_data_dir] --test_path [test_data_dir]

If you use the preprocessed data from <a href="https://github.com/fangchangma/sparse-to-dense.pytorch">here</a>. The train/test data path should be ./kitti/train or ./kitti/val/ under your data directory.

Other specifications: `--continue_train` would load the lastest saved ckpt. Set hyperparameters by `--lr`, `--batch_size`, `--weight_decay`, or others. Please refer to the options/base_options.py and options/options.py

Example command:

	python3 train_depth_complete.py --name kitti --checkpoints_dir ./checkpoints --lr 0.001 --batch_size 4 --train_path './kitti/train/' --test_path './kitti/val/' --continue_train
	
For evalutation, please run

	python3 evaluate.py --name kitti --checkpoints_dir [path to save_dir to load ckpt] --test_path [test_data_dir] [--epoch [epoch number]]

This will load the latest checkpoint to evaluate. Add `--epoch` to specify which epoch checkpoint you want to load.

If you find our work useful, please consider to cite our work.
`
@article{zhong2019deep,
  title={Deep RGB-D Canonical Correlation Analysis For Sparse Depth Completion},
  author={Zhong, Yiqi and Wu, Cho-Ying and You, Suya and Neumann, Ulrich},
  journal={arXiv preprint arXiv:1906.08967},
  year={2019}
}`

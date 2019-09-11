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
Train/Test
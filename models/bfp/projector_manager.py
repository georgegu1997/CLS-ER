# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import MNISTMLP
from utils.args import *
from utils.lowrank_reg import LowRankReg
from utils.routines import forward_loader_all_layers

from .utils import *
from .linear_decomposed import LinearDecomposed

def add_parser(parser):
	parser.add_argument('--alpha_bfp', type=float, required=True,
				help="Weight of the backward feature projection loss. It can be overridden by the 'alpha_bfpX' below")
	parser.add_argument('--alpha_bfp1', type=float, default=None)
	parser.add_argument('--alpha_bfp2', type=float, default=None)
	parser.add_argument('--alpha_bfp3', type=float, default=None)
	parser.add_argument('--alpha_bfp4', type=float, default=None)

	parser.add_argument('--alpha_lr', type=float, default=None,
				help='Weight of the low-rank regularization loss.')

	parser.add_argument("--alpha_svd_sep", type=float, default=1,
				help="Weight of the singular direction separation loss.")

	parser.add_argument("--alpha_sim", type=float, default=0,
				help="Weight of the similarity preserving loss (applied only on old data).")

	parser.add_argument('--loss_type', type=str, default='mfro', choices=['mse', 'rmse', 'mfro', 'cos'],
				help='How to compute the matching loss on projected features.')
	parser.add_argument("--normalize_feat", action="store_true",
				help="if set, normalize features before computing the matching loss.")
	parser.add_argument("--opt_type", type=str, default="sgdm", choices=["sgd", "sgdm", "adam"],
				help="Optimizer type.")
	parser.add_argument("--proj_lr", type=float, default=0.1,
				help="Learning rate for the optimizer on the projectors.")    
	parser.add_argument("--momentum", type=float, default=0.9,
				help="Momentum for SGD.")

	parser.add_argument('--proj_init_identity', action="store_true",
				help="If set, initialize the projectors to the identity mapping.")
	parser.add_argument('--proj_task_reset', type=str2bool, default=True,
				help="If set, initialize the projectors to a random mapping.")

	parser.add_argument('--proj_type', type=str, default="1", choices=['0', '1', '2', '0p+1'],
				help="Type of the backward feature projection. (number of layers in MLP projector)")
	parser.add_argument('--final_feat', action='store_true',
				help="If true, bfp loss will only be applied to the last feature map.")
	parser.add_argument('--pool_dim', default='hw', type=str, choices=['h', 'w', 'c', 'hw', 'flatten'], 
				help="Pooling before computing BFP loss. If None, no pooling is applied.")

	parser.add_argument('--decompose_proj', action='store_true',
				help="If true, the LinearDecomposed will be used for projector.")
	parser.add_argument('--alpha_proj_S_reg', type=float, default=0, 
				help='Weight of the projection matrix S regularization loss.')
	parser.add_argument('--alpha_proj_V_reg', type=float, default=0,
				help='Weight of the projection matrix V regularization loss.')
				
	parser.add_argument('--proj_feat_svd', action='store_true',
				help="If true, the features will be projected onto most prominent singular vectors, before computing bfp loss.")
	parser.add_argument('--proj_feat_svd_n_basis_factor', type=float, default=1,
				help="Number of basis = Total classes seen + proj_feat_svd_n_basis_factor")
	parser.add_argument('--svd_sep', action='store_true',
				help="If true, the features of new data will be enforced to be orthogonal to old singular vectors.")
	
	return parser

class ProjectorManager(nn.Module):
	'''
	Helper class managing the projection layers for BFP
	Such that it can be easily integrated into other continual learning methods
	'''
	def __init__(self, args, net_channels, device):
		super(ProjectorManager, self).__init__()
		self.args = args
		self.net_channels = net_channels
		self.device = device
		
		# Initialize the backward projection layers
		self.alpha_bfp_list = [self.args.alpha_bfp] * len(self.net_channels)
		if self.args.alpha_bfp1 is not None: self.alpha_bfp_list[0] = self.args.alpha_bfp1
		if self.args.alpha_bfp2 is not None: self.alpha_bfp_list[1] = self.args.alpha_bfp2
		if self.args.alpha_bfp3 is not None: self.alpha_bfp_list[2] = self.args.alpha_bfp3
		if self.args.alpha_bfp4 is not None: self.alpha_bfp_list[3] = self.args.alpha_bfp4
		self.bfp_flag = sum(self.alpha_bfp_list) > 0
		
		self.reset_proj()
		
		# Initialize the low-rank regularization layer, applied on the BFP transformation matrix
		if self.args.alpha_lr:
			final_dim = self.net_channels[-1] 
			self.lr_regularizer = LowRankReg(final_dim, lr=1e-3, rank=final_dim // self.args.N_TASKS)
			self.lr_regularizer.to(self.device)

		# Determine whether to compute the SVD of the features
		if self.args.decompose_proj or self.args.svd_sep or \
			self.args.proj_feat_svd or 'p' in self.args.proj_type:
			self.compute_svd = True
		else:
			self.compute_svd = False

		# Statistics (feature distribution under PCA) of the past data
		self.U, self.S, self.V = None, None, None
		self.feat_mean = None

		# Get the list of layers where BFP is applied
		if self.args.final_feat:
			self.layers_bfp = [-1]
		else:
			self.layers_bfp = list(range(len(self.net_channels)))

	def begin_task(self, dataset, t=0, start_epoch=0):
		self.task_id = t
		if not self.bfp_flag: return

		if self.args.alpha_lr:
			rank = self.net_channels[-1] // self.args.N_TASKS * (t+1)
			print("Set lr_regularizer rank to {}".format(rank))
			self.lr_regularizer.set_rank(rank)

		if self.args.proj_task_reset:
			self.reset_proj()


	def _get_projector(self, feat_dim, init_identity=False):
		if self.args.decompose_proj:
			projector = LinearDecomposed(feat_dim)

			# Initialize the projector according to statstics of the past data
			if self.V is not None and feat_dim == self.V.shape[0]:
				projector.set_parameters(self.V.T, self.feat_mean, init_identity)
		elif self.args.proj_type == '0':
			projector = nn.Identity()
		elif '1' in self.args.proj_type:
			projector = nn.Linear(feat_dim, feat_dim)
			if init_identity:
				projector.weight.data = torch.eye(feat_dim)
				projector.bias.data = torch.zeros(feat_dim)
		elif '2' in self.args.proj_type:
			projector = nn.Sequential(
				nn.Linear(feat_dim, feat_dim),
				nn.ReLU(),
				nn.Linear(feat_dim, feat_dim),
			)
		else:
			raise Exception("Unknown projector type: {}".format(self.args.proj_type))

		projector.to(self.device)
		return projector
		
	def reset_proj(self):
		# Get one optimizer for each network layer
		self.projectors = nn.ModuleList()
		for c in self.net_channels:
			projector = self._get_projector(c, self.args.proj_init_identity)
			self.projectors.append(projector)

		if self.args.proj_type != '0':
			# Optimizer for all projectors
			if self.args.opt_type == 'sgd':
				self.opt_proj = SGD(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr)
			elif self.args.opt_type == 'sgdm':
				self.opt_proj = SGD(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr, momentum=self.args.momentum)
			elif self.args.opt_type == 'adam':
				self.opt_proj = Adam(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr)
		else:
			self.opt_proj = None

	def proj_feat_V(self, feat):
		'''
		Project the feature along the first n_basis vectors of V
		'''
		assert self.V is not None
		# Number of basis = the number of classes seen up to the last task
		n_basis = self.task_id * self.args.N_CLASSES_PER_TASK * self.args.proj_feat_svd_n_basis_factor
		n_basis = int(round(n_basis))
		basis = self.V[:, :n_basis]
		feat = (feat - self.feat_mean) @ basis @ basis.T + self.feat_mean
		return feat

	def svd_sep_loss(self, feat):
		'''
		Loss that forces the feature to be learned outside the subspace occupied by old classes
		'''
		n_basis = self.task_id * self.args.N_CLASSES_PER_TASK
		basis = self.V[:, :n_basis]
		feat = (feat - self.feat_mean) @ basis @ basis.T + self.feat_mean
		loss = match_loss(feat, torch.zeros_like(feat), self.args.loss_type)
		return loss

	def sim_loss(self, f1, f2):
		'''
		Loss that preserves the self-inner product of two features
		'''
		# sim1 = torch.mm(f1, f1.T) # (n_old, n_old)
		# sim2 = torch.mm(f2, f2.T) # (n_old, n_old)

		# Compute the similarity matrix as the pairwise euclidean distance
		sim1 = torch.cdist(f1, f1) # (n_old, n_old)
		sim2 = torch.cdist(f2, f2) # (n_old, n_old)
		
		# Row-wise normalization
		sim1 = sim1 / sim1.sum(dim=1, keepdim=True)
		sim2 = sim2 / sim2.sum(dim=1, keepdim=True)
		
		sim_loss = self.args.alpha_sim * F.mse_loss(sim2, sim1)
		return sim_loss

	def compute_loss(self, feats, feats_old, mask_new, mask_old):
		bfp_loss = 0.0
		lr_loss = 0.0
		proj_S_reg_loss = 0.0
		proj_V_reg_loss = 0.0
		svd_sep_loss = 0.0
		sim_loss = 0.0

		for i in self.layers_bfp:
			projector = self.projectors[i]
			feat = feats[i]
			feat_old = feats_old[i]
			
			# After pooling, feat and feat_old have shape (n, d)
			feat, feat_old = pool_feat(feat, feat_old, self.args.pool_dim, self.args.normalize_feat)

			# loss that enforces new features to be far from old singular vectors
			# Only applied on the data of the new tasks
			if self.args.svd_sep:
				feat_nn = feat[mask_new] # feat of new data, new model
				svd_sep_loss += self.svd_sep_loss(feat_nn)

			# Similarity preserving loss between the old and new features
			# Now only applied on the old data from memory
			if self.args.alpha_sim:
				sim_loss += self.sim_loss(feat_old[mask_old], feat[mask_old])

			# Project the features onto subspace spanned by the old features
			if self.args.proj_feat_svd:
				feat = self.proj_feat_V(feat)
				feat_old = self.proj_feat_V(feat_old)

			feat_proj = projector(feat) # (N, C)
			bfp_loss += self.alpha_bfp_list[i] * match_loss(feat_proj, feat_old, self.args.loss_type)

			if '0p+' in self.args.proj_type:
				# use match loss on feat and feat_old after self.proj_feat_V in addition to the bfp loss
				bfp_loss += self.alpha_bfp_list[i] * match_loss(
					self.proj_feat_V(feat), self.proj_feat_V(feat_old), self.args.loss_type)

			if self.args.decompose_proj:
				w = self.S.sqrt()
				w = w / w.max()
				loss_S, loss_V = projector.soft_reg_loss(w)
				proj_S_reg_loss += self.args.alpha_proj_S_reg * loss_S
				proj_V_reg_loss += self.args.alpha_proj_V_reg * loss_V

		bfp_loss /= len(self.layers_bfp)
		proj_S_reg_loss /= len(self.layers_bfp)
		proj_V_reg_loss /= len(self.layers_bfp)
		svd_sep_loss /= len(self.layers_bfp)
		sim_loss /= len(self.layers_bfp)

		'''Low-rank regularization'''
		if self.args.alpha_lr:
			feat_final = feats[-1]
			feat_final = F.avg_pool2d(feat_final, feat_final.shape[2]).view(feat_final.shape[0], -1)
			feat_final_lr, loss = self.lr_regularizer.forward_and_loss(feat_final)
			lr_loss = self.args.alpha_lr * loss

		loss = bfp_loss + lr_loss + proj_S_reg_loss + proj_V_reg_loss + svd_sep_loss + sim_loss

		loss_dict = {
			'match_loss': bfp_loss,
			'lr_loss': lr_loss,
			'proj_S_reg_loss': proj_S_reg_loss,
			'proj_V_reg_loss': proj_V_reg_loss,
			'svd_sep_loss': svd_sep_loss,
			'sim_loss': sim_loss,
		}

		return loss, loss_dict

	def compute_feat_statistics(self, net, loader):
		logits, feats, xs, ys = forward_loader_all_layers(net, loader)
		feats, _ = pool_feat(feats[-1], feats[-1], self.args.pool_dim, self.args.normalize_feat)

		self.feat_mean = feats.mean(0).unsqueeze(0)
		U, S, V = torch.svd(feats - self.feat_mean, compute_uv=True)
		self.U, self.S, self.V = U.to(self.device), S.to(self.device), V.to(self.device)
		self.feat_mean = self.feat_mean.to(self.device)

	def before_backward(self):
		if not self.bfp_flag: return

		if self.opt_proj is not None: self.opt_proj.zero_grad()
		if self.args.alpha_lr: self.lr_regularizer.zero_grad()

	def end_task(self, dataset, net):
		if not self.bfp_flag: return

		if self.args.decompose_proj:
			# Visualize and plot the projection matrix
			projector = self.projectors[-1]
			x_values = np.arange(len(projector.S))
			y_values = projector.S.detach().cpu().numpy()
			data = [[x, y] for (x, y) in zip(x_values, y_values)]
			table = wandb.Table(data=data, columns = ["S-index", "S-value"])
			wandb.log({"proj/proj_S" : wandb.plot.line(table, "S-index", "S-value",
					title="singular values of the projection matrix")})

			figure = plt.figure()
			V = projector.V.detach().cpu().numpy()
			V_inner = V.T @ V
			plt.imshow(V_inner)
			plt.colorbar()
			wandb.log({"proj/proj_V_inner" : figure})

			figure = plt.figure()
			U = projector.U.detach().cpu().numpy()
			U_inner = U.T @ U
			plt.imshow(U_inner)
			plt.colorbar()
			wandb.log({"proj/proj_U_inner" : figure})

			U_V_cross = U.T @ V
			figure = plt.figure()
			plt.imshow(U_V_cross)
			plt.colorbar()
			wandb.log({"proj/proj_U_V_cross" : figure})

		if self.compute_svd:
			# Compute the SVD statistics for the current training set
			with torch.no_grad():
				loader = dataset.train_loaders[self.task_id]
				self.compute_feat_statistics(net, loader)

	def step(self):
		if not self.bfp_flag: return

		if self.opt_proj is not None: self.opt_proj.step()

		# For low-rank regularization
		if self.args.alpha_lr: 
			self.lr_regularizer.step()
			self.lr_regularizer.SVP()
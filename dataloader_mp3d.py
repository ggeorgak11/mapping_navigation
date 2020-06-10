from torch.utils.data import Dataset
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from parameters_mp3d import ParametersMapNet_MP3D, ParametersIL_MP3D, ParametersQNet_MP3D
import cv2
from PIL import Image 
import data_helper_mp3d as dhm
import torch
import gzip
import json
import networkx as nx

# MP3D class for online sampling of episodes.
class MP3D_online(Dataset):
	def __init__(self, par, nStartPos, scene_list, action_list, init_configs=None, graphs_dict=None):
		self.datasetPath = par.mp3d_root
		self.cropSize = par.crop_size
		self.cropSizeObsv = par.crop_size_obsv
		self.orig_res = par.orig_res
		self.normalize = True
		self.pixFormat = "NCHW"
		self.scene_list = scene_list #['Home_001_1']
		self.n_start_pos = nStartPos # how many episodes to sample per scene
		self.max_shortest_path = par.max_shortest_path
		self.stop_on_collision = par.stop_on_collision
		self.actions = action_list
		self.dets_nClasses = par.dets_nClasses
		# Read the semantic segmentations
		self.cat_dict = par.cat_dict # category names to labels dictionary
		#self.max_steps = par.max_steps

		# Need to collect the image names for the goals (read arsalan's file)
		targets_file_path = '{}/{}.npy'.format(self.datasetPath, 'annotated_targets')
		self.targets_data = np.load(targets_file_path, allow_pickle=True).item()
		self.detection_data = dhm.load_detections(self.datasetPath, self.scene_list)
		#print(data.keys()) # [b'tv', b'dining_table', b'fridge', b'microwave', b'couch']
		#print(data[b'fridge'].keys())

		# Need to pre-calculate the graphs of the scenes for every target
		self.graphs_dict = graphs_dict
		if graphs_dict is None:
			self.graphs_dict = dhm.get_scene_target_graphs(self.datasetPath, self.cat_dict, self.targets_data, self.actions)
		
		if init_configs is None:
			# Randomly sample the starting points for all episodes. 
			self.sample_starting_points()
		else: # use the prespecified init configurations to get the samples
			self.get_predefined_inits(init_configs)

	def get_predefined_inits(self, init_configs):
		start_id=0
		images, scene_infos, scene_name, scene_scale, targets= {}, {}, {}, {}, {}
		for scene in self.scene_list:
			scene_folder = '{}/{}'.format(self.datasetPath, scene)
			info = dhm.load_scene_info(scene_folder)
			scene_infos[scene] = info
			scene_configs = init_configs[scene]
			# get the objects present in each scene
			categories = dhm.candidate_targets(scene, self.cat_dict, self.targets_data)
			for i in range(len(scene_configs)):
				im_name_0 = scene_configs[i][0]
				cat = scene_configs[i][1]
				# for every starting position create an example using all available categories
				#for cat in categories:
				target_lbl = self.cat_dict[cat]
				
				graph = self.graphs_dict[target_lbl][scene]
				path = nx.shortest_path(graph, im_name_0, "goal")
				if len(path) > 20:
					continue
				print(start_id, im_name_0, cat, len(path))
				
				images[start_id] = im_name_0
				scene_name[start_id] = scene
				scene_scale[start_id] = 1.0 #scale
				targets[start_id] = target_lbl # get target lbl
				start_id += 1
		#raise Exception("111")
		self.scene_infos = scene_infos
		self.images = images
		self.scene_name = scene_name
		self.scene_scale = scene_scale
		self.targets = targets

	def sample_starting_points(self):
		# Store all info necessary for a starting position
		start_id=0
		images, scene_infos, scene_name, scene_scale, targets, pose = {}, {}, {}, {}, {}, {}
		for scene in self.scene_list:
			scene_folder = '{}/{}'.format(self.datasetPath, scene)
			info = dhm.load_scene_info(scene_folder)
			im_names_all = list(info.keys()) # info 0 # list of image names in the scene
			im_names_all = np.hstack(im_names_all) # flatten the array
			n_images = len(im_names_all)
			# get poses for visualization purposes
			#world_poses = info['world_pos'] # info 3
			#directions = info['direction'] # info 4
			scene_start_count = 0
			while scene_start_count < self.n_start_pos:
				
				#im_name_0 = "000110009350101.jpg" # *** Hardcode the episode for now ***
				# Randomly select a target that exists in that scene
				# Get the list of possible targets
				candidates = dhm.candidate_targets(scene, self.cat_dict, self.targets_data)
				idx_cat = np.random.randint(len(candidates), size=1)
				cat = candidates[idx_cat[0]]
				#print(cat)
				target_lbl = self.cat_dict[cat]
				#target_lbl = 5 # *** Hardcode the episode for now ***
				graph = self.graphs_dict[target_lbl][scene]
				# Randomly select an image index as the starting position
				while True:
					# Randomly select an image index as the starting position	  
					idx = np.random.randint(n_images, size=1)
					im_name_0 = im_names_all[idx[0]]
					if dhm.check_if_goal_is_reachable(im_name_0, graph):
						break
				path = nx.shortest_path(graph, im_name_0, "goal")
				##if len(path)-1 == 5:
				##	print(im_name_0, target_lbl)
				##	print(path)
				#	# get set of actions in that path that lead to goal
				##	act = []
				##	for j in range(len(path)-1):
				##		act.append(graph.edges[path[j], path[j+1]]['action'])
				##	print(act)
				##	raise Exception("777")
				if len(path)-2 == 0: # this means that im_name_0 is a goal location
					continue
				if len(path)-1 > self.max_shortest_path: # limit on the length of episodes
					continue
				#print(len(path))
				# get poses for visualization purposes
				#print(world_poses[idx[0]])
				#pos_tmp = world_poses[idx[0]] * scale # 3 x 1
				#print(pos_tmp)
				#pose_x_gt = pos_tmp[0,:]
				#pose_z_gt = pos_tmp[2,:]
				#dir_tmp = directions[idx[0]] # 3 x 1
				#dir_gt = np.arctan2(dir_tmp[2,:], dir_tmp[0,:])[0] # [-pi,pi]
				###poses_epi.append([pose_x_gt, pose_z_gt, dir_gt])	

				# Add the starting location in the pool
				images[start_id] = im_name_0
				scene_name[start_id] = scene
				scene_scale[start_id] = 1.0 #scale
				targets[start_id] = target_lbl #self.cat_dict[cat]
				#pose[start_id] = np.array([pose_x_gt, pose_z_gt, dir_gt], dtype=np.float32)
				scene_start_count += 1
				start_id += 1
			scene_infos[scene] = info
		self.scene_infos = scene_infos
		self.images = images
		self.scene_name = scene_name
		self.scene_scale = scene_scale
		self.targets = targets
		#self.pose = pose

	# returns starting points
	def __getitem__(self, index):
		item = {}
		im_name = self.images[index]
		scene = self.scene_name[index]
		scale = self.scene_scale[index]
		target_lbl = self.targets[index]
		#pose = self.pose[index]

		imgData, im_sseg, im_dets, points2D, local3D = dhm.getImageData(self.datasetPath, self.detection_data, self.dets_nClasses,im_name, 
			scene, scale, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
		im_obsv, im_dets_obsv = dhm.getImageData(self.datasetPath, self.detection_data, 1, im_name, scene, scale, 
			self.cropSizeObsv, self.orig_res, self.pixFormat, self.normalize, get3d=False)

		item["image"] = torch.from_numpy(imgData).float() # 3 x h x w
		#item["image"] = imgData
		item["image_name"] = im_name
		item["points2D"] = points2D # n_points x 2
		item["local3D"] = local3D # n_points x 3
		item["scene"] = scene
		item["scale"] = scale
		item['sseg'] = torch.from_numpy(im_sseg).float() # 1 x h x w
		item['dets'] = torch.from_numpy(im_dets).float() # 91 x h x w
		item['image_obsv'] = torch.from_numpy(im_obsv).float()
		item['dets_obsv'] = torch.from_numpy(im_dets_obsv).float()
		#item['sseg'] = im_sseg
		item['target_lbl'] = target_lbl
		return item

	def __len__(self):
		return len(self.images)

	# given the next image retrieve the relevant info
	# assumes that next_im is not a collision
	#def get_step_data(self, actions, im_names, scenes, scales):
	def get_step_data(self, next_ims, scenes, scales):
		batch_size = len(next_ims)
		imgs_batch_next = torch.zeros(batch_size, 3, self.cropSize[1], self.cropSize[0]).float().cuda()
		sseg_batch_next = torch.zeros(batch_size, 1, self.cropSize[1], self.cropSize[0]).float().cuda()
		dets_batch_next = torch.zeros(batch_size, self.dets_nClasses, self.cropSize[1], self.cropSize[0]).float().cuda()
		imgs_obsv_batch_next = torch.zeros(batch_size, 3, self.cropSizeObsv[1], self.cropSizeObsv[0]).float().cuda()
		dets_obsv_batch_next = torch.zeros(batch_size, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]).float().cuda()
		points2D_batch_next, local3D_batch_next = [], []
		for b in range(batch_size):
			next_im = next_ims[b]
			scene = scenes[b]
			scale = scales[b]

			imgData, im_sseg, im_dets, points2D, local3D = dhm.getImageData(self.datasetPath,self.detection_data, self.dets_nClasses, next_im, 
					scene, scale, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
			im_obsv, im_dets_obsv = dhm.getImageData(self.datasetPath,self.detection_data, 1, next_im, scene, scale, self.cropSizeObsv, 
				self.orig_res, self.pixFormat, self.normalize, get3d=False)
			
			imgs_batch_next[b,:,:,:] = torch.from_numpy(imgData).float()
			sseg_batch_next[b,:,:,:] = torch.from_numpy(im_sseg).float()
			dets_batch_next[b,:,:,:] = torch.from_numpy(im_dets).float()
			imgs_obsv_batch_next[b,:,:,:] = torch.from_numpy(im_obsv).float()
			dets_obsv_batch_next[b,:,:,:] = torch.from_numpy(im_dets_obsv).float()
			points2D_batch_next.append(points2D)
			local3D_batch_next.append(local3D)
			mapNet_batch = (imgs_batch_next, points2D_batch_next, local3D_batch_next, sseg_batch_next, dets_batch_next)
			obsv_batch = (imgs_obsv_batch_next, dets_obsv_batch_next)
		return mapNet_batch, obsv_batch


# Tailored to work with mapNet training
class MP3D(Dataset):

	def __init__(self, par, seq_len, nEpisodes, scene_list, action_list, with_shortest_path=False):
		self.datasetPath = par.mp3d_root
		self.cropSize = par.crop_size
		self.orig_res = par.orig_res
		self.normalize = True
		self.pixFormat = 'NCHW'
		self.scene_list = scene_list
		self.n_episodes = nEpisodes # how many episodes to sample per scene
		self.seq_len = seq_len # used when training MapNet with constant sequence length
		self.actions = action_list
		self.dets_nClasses = par.dets_nClasses
		self.detection_data = dhm.load_detections(self.datasetPath, self.scene_list)
		#self.sample_episodes()
		self.flag_shortest_path = with_shortest_path # if True, use shortest path, otherwise, randomize the trajectory.
		if with_shortest_path:
			self.sample_episodes()
		else:
			self.sample_episodes_random()

	# sample episodes from each scene and write them to a temporary json.gz file
	def sample_episodes(self):
		# This function chooses the actions through shortest path
		epi_id=0 # episode id
		im_paths, pose, scene_name, scene_scale = {}, {}, {}, {}

		for scene in self.scene_list:
			scene_folder = '{}/{}'.format(self.datasetPath, scene)
			info = dhm.load_scene_info(scene_folder)
			# annotations contains the action information
			# info contains camera information for each image
			im_names_all = list(info.keys()) # list of image names in the scene
			im_names_all = np.hstack(im_names_all) # flatten the array
			n_images = len(im_names_all)
			
			# Create the graph of the environment
			graph = dhm.create_scene_graph(info, im_names=im_names_all, action_set=self.actions)

			scene_epi_count = 0
			while scene_epi_count < self.n_episodes:
				# Randomly select an image index as the starting position
				# Need to do the shortest path between two locations otherwise the episode might be only rotating around

				# Randomly select two nodes and sample a trajectory across their shortest path
				idx = np.random.randint(n_images, size=2)
				im_name_0 = im_names_all[idx[0]]
				im_name_1 = im_names_all[idx[1]]
				#print(im_name_0, im_name_1)
				
				# organize the episodes into dictionaries holding different information
				# i.e. episode_id : list of values (img path, actions, world poses, directions)
				if nx.has_path(graph, im_name_0, im_name_1):
					path = nx.shortest_path(graph, im_name_0, im_name_1) # sequence of nodes leading to goal
					#print('path = {}'.format(path))
					#print(len(path))
					if len(path) >= self.seq_len:
						#print(path)
						# get set of actions in that path that lead from image_0 to image_1
						#act = []
						#for j in range(len(path)-1):
						#	act.append(graph.edges[path[j], path[j+1]]['action'])
						#print(act)
						#poses_epi, dir_epi = [], [] # hold the world positions and directions for this episode
						poses_epi = []
						for i in range(self.seq_len):
							next_im = path[i]
							poses_epi.append(info[next_im]['pose'])

						im_paths[epi_id] = np.asarray(path[:self.seq_len])
						pose[epi_id] = np.asarray(poses_epi, dtype=np.float32)
						scene_name[epi_id] = scene
						scene_scale[epi_id] = 1.0 #scale
						epi_id+=1
						scene_epi_count+=1
						
		self.im_paths = im_paths
		self.pose = pose
		self.scene_name = scene_name
		self.scene_scale = scene_scale

	def sample_episodes_random(self):
		# This function chooses the actions randomly and not through shortest path
		epi_id=0 # episode id
		im_paths, pose, scene_name, scene_scale = {}, {}, {}, {}
		#acts = {}
		for scene in self.scene_list:
			scene_folder = '{}/{}'.format(self.datasetPath, scene)
			info = dhm.load_scene_info(scene_folder)
			# info contains pose and action for each image
			im_names_all = list(info.keys()) # list of image names in the scene
			im_names_all = np.hstack(im_names_all) # flatten the array
			n_images = len(im_names_all)

			scene_epi_count = 0
			while scene_epi_count < self.n_episodes:
				# Randomly select an image index as the starting position
				idx = np.random.randint(n_images, size=1)
				im_name_0 = im_names_all[idx[0]]		  
				# organize the episodes into dictionaries holding different information
				poses_epi, path = [], []
				for i in range(self.seq_len):
					if i==0:
						current_im = im_name_0
					else:
						# randomly choose the action
						sel_action = self.actions[np.random.randint(len(self.actions), size=1)[0]]
						next_im = info[current_im][sel_action]
						if not(next_im==''):
							current_im = next_im
					path.append(current_im)
					poses_epi.append(info[current_im]['pose'])
				im_paths[epi_id] = np.asarray(path)
				pose[epi_id] = np.asarray(poses_epi, dtype=np.float32)
				scene_name[epi_id] = scene
				scene_scale[epi_id] = 1.0 #scale
				epi_id+=1
				scene_epi_count+=1

		self.im_paths = im_paths
		self.pose = pose
		self.scene_name = scene_name
		self.scene_scale = scene_scale

	def __len__(self):
		return len(self.im_paths)

	# Return the index episode
	def __getitem__(self, index):
		item = {}
		path = self.im_paths[index]
		poses_epi = self.pose[index]
		scene = self.scene_name[index]
		scale = self.scene_scale[index]

		# points2D and local3D for each step have different sizes, so save them in dictionaries:
		# key:value ==> step:points2D/local3D
		imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		points2D, local3D = [], []
		for i in range(len(path)): # seq_len
			im_name = path[i]
			imgData, ssegData, detData, points2D_step, local3D_step = dhm.getImageData(self.datasetPath, self.detection_data, self.dets_nClasses,im_name, 
														scene, scale, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
			#print(imgData.shape)
			#print(points2D_step.shape)
			#print(local3D_step.shape)
			imgs[i,:,:,:] = imgData
			#points2D[i] = points2D_step
			#local3D[i] = local3D_step
			points2D.append(points2D_step) 
			local3D.append(local3D_step)
			# Load the semantic segmentations
			#ssegs[i,:,:,:] = dh.get_sseg(im_name, scene_seg, self.cropSize)
			#dets[i,:,:,:] = dh.get_det_mask(im_name, scene_dets, self.cropSize, self.dets_nClasses, self.labels_to_index)
			ssegs[i, :, :, :] = ssegData
			dets[i, :, :, :] = detData

		# Need to get the relative poses (towards the first frame) for the ground-truth
		rel_poses = dhm.relative_poses(poses=poses_epi)

		item["images"] = torch.from_numpy(imgs).float()
		item["images_names"] = path
		item["points2D"] = points2D # nested list of seq_len x n_points x 2
		item["local3D"] = local3D # nested list of seq_len x n_points x 3
		item["pose"] = rel_poses
		item["abs_pose"] = poses_epi
		item["scene"] = scene
		item["scale"] = scale
		item['sseg'] = torch.from_numpy(ssegs).float()
		item['dets'] = torch.from_numpy(dets).float()
		return item

class MP3D_IL(Dataset):

	def __init__(self, par, seq_len, nEpisodes, scene_list, action_list):
		self.datasetPath = par.mp3d_root
		self.cropSize = par.crop_size
		self.cropSizeObsv = par.crop_size_obsv
		self.orig_res = par.orig_res
		self.normalize = True
		self.pixFormat = 'NCHW'
		self.scene_list = scene_list
		self.n_episodes = nEpisodes # how many episodes to sample per scene
		self.seq_len = seq_len # used when training MapNet with constant sequence length
		self.actions = action_list
		self.dets_nClasses = par.dets_nClasses
		# target category names to labels dictionary
		self.cat_dict = par.cat_dict
		# Need to collect the image names for the goals
		targets_file_path = '{}/{}.npy'.format(self.datasetPath, 'annotated_targets')
		self.targets_data = np.load(targets_file_path, allow_pickle=True).item()
		self.detection_data = dhm.load_detections(self.datasetPath, self.scene_list)
		self.graphs_dict = dhm.get_scene_target_graphs(self.datasetPath, self.cat_dict, self.targets_data, self.actions)
		self.sample_episodes()

	def sample_episodes(self):
		# Each episode should contain:
		# List of images, list of actions, cost of every action, scene, scale, maybe collision and previous action indicators?
		epi_id=0 # episode id
		im_paths, action_paths, cost_paths, scene_name, scene_scale, target_lbls, pose_paths, collisions = {}, {}, {}, {}, {}, {}, {}, {}
		for scene in self.scene_list:
			#print(scene)
			scene_folder = '{}/{}'.format(self.datasetPath, scene)
			info = dhm.load_scene_info(scene_folder)
			# annotations contains the action information
			# info contains camera information for each image
			im_names_all = list(info.keys()) # info 0 # list of image names in the scene
			im_names_all = np.hstack(im_names_all) # flatten the array
			n_images = len(im_names_all)

			scene_epi_count = 0
			while scene_epi_count < self.n_episodes:
				# Randomly select a target that exists in that scene
				# Get the list of possible targets
				candidates = dhm.candidate_targets(scene, self.cat_dict, self.targets_data)
				idx_cat = np.random.randint(len(candidates), size=1)
				cat = candidates[idx_cat[0]]
				target_lbl = self.cat_dict[cat]
				graph = self.graphs_dict[target_lbl][scene] # to be used to get the ground-truth
				#path = nx.shortest_path(graph, im_name_0, "goal") 
				# check if episode has at least n steps...?
				# Use the im_name_0 as a starting position and randomly choose actions for max steps
				# For each action, use the graph to estimate its cost
				#path_dist_init = len(nx.shortest_path(graph, im_name_0, "goal"))-2 # get cost of first image
				#if path_dist_init < self.seq_len: # ** check this again 
				#	continue

				# Choose whether the episode's observations are going to be decided by the
				# teacher (best action) or randomly
				choice = np.random.randint(2, size=1)[0] # if 1 then do teacher
				#choice = 1

				im_seq, action_seq, cost_seq, poses_seq, collision_seq = [], [], [], [], []
				while True:
					# Randomly select an image index as the starting position	  
					idx = np.random.randint(n_images, size=1)
					im_name_0 = im_names_all[idx[0]]
					if dhm.check_if_goal_is_reachable(im_name_0, graph):
						break
				im_seq.append(im_name_0)
				#cost_seq.append(len(path_dist_init))
				current_im = im_name_0
				# get the ground-truth cost for each next state
				cost_seq.append(dhm.get_state_action_cost(current_im, self.actions, info, graph))
				poses_seq.append(info[current_im]['pose'])
				collision_seq.append(0)
				for i in range(1, self.seq_len):
					# either select the best action or ...				
					# ... randomly choose the next action to move in the episode
					if choice:
						actions_cost = np.array(cost_seq[i-1])
						min_cost = np.min(actions_cost)
						min_ind = np.where(actions_cost==min_cost)[0]
						if len(min_ind)==1:
							sel_ind = min_ind[0]
						else: # if multiple actions have the lowest value then randomly select one
							sel_ind = min_ind[np.random.randint(len(min_ind), size=1)[0]]
						sel_action = self.actions[sel_ind]
					else:
						sel_action = self.actions[np.random.randint(len(self.actions), size=1)[0]]
					next_im = info[current_im][sel_action]
					#print(current_im, sel_action, next_im)
					if not (next_im==''): # if there is a collision then keep the same image
						current_im = next_im
						collision_seq.append(0)
					else:
						collision_seq.append(1)
					im_seq.append(current_im)
					action_seq.append(sel_action)
					# get the ground-truth pose
					poses_seq.append(info[current_im]['pose'])
					cost_seq.append(dhm.get_state_action_cost(current_im, self.actions, info, graph))

				im_paths[epi_id] = np.asarray(im_seq)
				action_paths[epi_id] = np.asarray(action_seq)
				cost_paths[epi_id] = np.asarray(cost_seq, dtype=np.float32)
				scene_name[epi_id] = scene
				scene_scale[epi_id] = 1.0 #scale
				target_lbls[epi_id] = target_lbl
				pose_paths[epi_id] = np.asarray(poses_seq, dtype=np.float32)
				collisions[epi_id] = np.asarray(collision_seq, dtype=np.float32)
				epi_id += 1
				scene_epi_count += 1

		self.im_paths = im_paths
		self.action_paths = action_paths
		self.cost_paths = cost_paths
		self.scene_name = scene_name
		self.scene_scale = scene_scale
		self.target_lbls = target_lbls
		self.pose_paths = pose_paths
		self.collisions = collisions

	def __len__(self):
		return len(self.im_paths)

	# Return the index episode
	def __getitem__(self, index):
		item = {}
		im_path = self.im_paths[index]
		action_path = self.action_paths[index]
		cost_path = self.cost_paths[index]
		scene = self.scene_name[index]
		scale = self.scene_scale[index]
		target_lbl = self.target_lbls[index]
		abs_poses = self.pose_paths[index]
		collision_seq = self.collisions[index]
		#scene_seg = self.sseg_data[scene.encode()] # convert string to byte
		#scene_dets = self.detection_data[scene]

		# points2D and local3D for each step have different sizes, so save them in dictionaries:
		# key:value ==> step:points2D/local3D
		imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		imgs_obsv = np.zeros((self.seq_len, 3, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
		ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		dets_obsv = np.zeros((self.seq_len, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
		points2D, local3D = [], [] #{}, {}
		for i in range(len(im_path)): # seq_len
			im_name = im_path[i]
			#print(index, i)
			imgData, ssegData, detData, points2D_step, local3D_step = dhm.getImageData(self.datasetPath,self.detection_data, self.dets_nClasses, im_name, 
														scene, scale, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
			imgs[i,:,:,:] = imgData
			points2D.append(points2D_step) 
			local3D.append(local3D_step)
			imgs_obsv[i,:,:,:], dets_obsv[i,:,:,:] = dhm.getImageData(self.datasetPath, self.detection_data, 1, im_name, scene, scale, self.cropSizeObsv, 
				self.orig_res, self.pixFormat, self.normalize, get3d=False)
			# Load the semantic segmentations
			#ssegs[i,:,:,:] = dh.get_sseg(im_name, scene_seg, self.cropSize)
			#dets[i,:,:,:] = dh.get_det_mask(im_name, scene_dets, self.cropSize, self.dets_nClasses, self.labels_to_index)
			#dets_obsv[i,:,:,:] = dh.get_det_mask(im_name, scene_dets, self.cropSizeObsv, 1, self.labels_to_index)
			ssegs[i, :, :, :] = ssegData
			dets[i, :, :, :] = detData

		# Need to get the relative poses (towards the first frame) for the ground-truth
		rel_poses = dhm.relative_poses(poses=abs_poses)

		item["images"] = torch.from_numpy(imgs).float()
		#item["images"] = imgs
		item["images_names"] = im_path
		item["points2D"] = points2D # nested list of seq_len x n_points x 2
		item["local3D"] = local3D # nested list of seq_len x n_points x 3
		item["actions"] = action_path
		item["costs"] = torch.from_numpy(cost_path).float()
		item["target_lbl"] = target_lbl
		item["pose"] = rel_poses
		item["abs_pose"] = abs_poses
		item["collisions"] = torch.from_numpy(collision_seq).float()
		item["scene"] = scene
		item["scale"] = scale
		item['sseg'] = torch.from_numpy(ssegs).float()
		#item['sseg'] = ssegs
		item['dets'] = torch.from_numpy(dets).float()
		item['images_obsv'] = torch.from_numpy(imgs_obsv).float()
		item['dets_obsv'] = torch.from_numpy(dets_obsv).float()
		return item







from torch.utils.data import Dataset, DataLoader
import os,glob,random,cv2,json,torch,cv2,h5py,math
import numpy as np
import networkx as nx
import data_helper as dh


class AVD_online(Dataset):
    # AVD class for online sampling of episodes.
    # Used primarily during testing.
    def __init__(self, par, nStartPos, scene_list, action_list, init_configs=None, graphs_dict=None):
        self.datasetPath = par.avd_root
        self.cropSize = par.crop_size
        self.cropSizeObsv = par.crop_size_obsv
        self.orig_res = par.orig_res
        self.normalize = True
        self.pixFormat = "NCHW"
        self.scene_list = scene_list
        self.n_start_pos = nStartPos # how many episodes to sample per scene
        self.max_shortest_path = par.max_shortest_path
        self.stop_on_collision = par.stop_on_collision
        self.actions = action_list
        self.dets_nClasses = par.dets_nClasses
        # Read the semantic segmentations
        self.sseg_data = np.load(par.sseg_file_path, encoding='bytes', allow_pickle=True).item()
        self.cat_dict = par.cat_dict # category names to labels dictionary
        # Load the detections, the object masks are created during getitem
        self.detection_data, self.labels_to_cats, self.labels_to_index = dh.load_detections(par, self.scene_list)
        # Need to collect the image names for the goals
        self.targets_data = np.load(par.targets_file_path, encoding='bytes', allow_pickle=True).item()
        # Need to pre-calculate the graphs of the scenes for every target
        self.graphs_dict = graphs_dict
        if graphs_dict is None:
            self.graphs_dict = dh.get_scene_target_graphs(self.datasetPath, self.cat_dict, self.targets_data, self.actions)
        
        if init_configs is None:
            # Randomly sample the starting points for all episodes. 
            self.sample_starting_points()
        else: # use the prespecified init configurations to get the samples
            self.get_predefined_inits(init_configs)


    def get_predefined_inits(self, init_configs):
        confs = np.load(init_configs) # [(scene, im_name, cat), (...), ...]
        start_id=0
        images, scene_annotations, scene_name, scene_scale, targets, path_length = {}, {}, {}, {}, {}, {}
        for conf in confs:
            (scene, im_name, cat) = conf
            annotations, scale, _, _, _ = dh.load_scene_info(self.datasetPath, scene)
            scene_annotations[scene] = annotations
            target_lbl = self.cat_dict[cat]
            graph = self.graphs_dict[target_lbl][scene]
            path = nx.shortest_path(graph, im_name, "goal")

            images[start_id] = im_name
            scene_name[start_id] = scene
            scene_scale[start_id] = scale
            targets[start_id] = target_lbl
            path_length[start_id] = len(path)
            start_id += 1

        self.scene_annotations = scene_annotations
        self.images = images
        self.scene_name = scene_name
        self.scene_scale = scene_scale
        self.targets = targets
        self.path_length = path_length


    def sample_starting_points(self):
        # Store all info necessary for a starting position
        start_id=0
        images, scene_annotations, scene_name, scene_scale, targets, pose, path_length = {}, {}, {}, {}, {}, {}, {}
        for scene in self.scene_list:
            annotations, scale, im_names_all, _, _ = dh.load_scene_info(self.datasetPath, scene)     
            scene_start_count = 0
            while scene_start_count < self.n_start_pos:
                # Randomly select an image index as the starting position
                idx = np.random.randint(len(im_names_all), size=1)
                im_name_0 = im_names_all[idx[0]]
                # Randomly select a target that exists in that scene
                candidates = dh.candidate_targets(scene, self.cat_dict, self.targets_data) # Get the list of possible targets
                idx_cat = np.random.randint(len(candidates), size=1)
                cat = candidates[idx_cat[0]]
                target_lbl = self.cat_dict[cat]
                graph = self.graphs_dict[target_lbl][scene]
                path = nx.shortest_path(graph, im_name_0, "goal")
                if len(path)-2 == 0: # this means that im_name_0 is a goal location
                    continue
                if len(path)-1 > self.max_shortest_path: # limit on the length of episodes
                    continue 
                # Add the starting location in the pool
                images[start_id] = im_name_0
                scene_name[start_id] = scene
                scene_scale[start_id] = scale
                targets[start_id] = target_lbl
                path_length[start_id] = len(path)
                scene_start_count += 1
                start_id += 1
            scene_annotations[scene] = annotations
        self.scene_annotations = scene_annotations
        self.images = images
        self.scene_name = scene_name
        self.scene_scale = scene_scale
        self.targets = targets
        self.path_length = path_length


    # returns starting points
    def __getitem__(self, index):
        item = {}
        im_name = self.images[index]
        scene = self.scene_name[index]
        scale = self.scene_scale[index]
        scene_seg = self.sseg_data[scene.encode()]
        target_lbl = self.targets[index]
        path_len = self.path_length[index]
        scene_dets = self.detection_data[scene]

        imgData, points2D, local3D = dh.getImageData(self.datasetPath, im_name, scene, 
                                                        self.cropSize, self.orig_res, self.pixFormat, self.normalize)
        im_sseg = dh.get_sseg(im_name, scene_seg, self.cropSize)
        im_dets = dh.get_det_mask(im_name, scene_dets, self.cropSize, self.dets_nClasses, self.labels_to_index)
        im_dets_obsv = dh.get_det_mask(im_name, scene_dets, self.cropSizeObsv, 1, self.labels_to_index)
        im_obsv = dh.getImageData(self.datasetPath, im_name, scene, self.cropSizeObsv, self.orig_res, 
                                                                        self.pixFormat, self.normalize, get3d=False)

        item["image"] = torch.from_numpy(imgData).float() # 3 x h x w
        item["image_name"] = im_name
        item["points2D"] = points2D # n_points x 2
        item["local3D"] = local3D # n_points x 3
        item["scene"] = scene
        item["scale"] = scale
        item['sseg'] = torch.from_numpy(im_sseg).float() # 1 x h x w
        item['dets'] = torch.from_numpy(im_dets).float() # 91 x h x w
        item['image_obsv'] = torch.from_numpy(im_obsv).float()
        item['dets_obsv'] = torch.from_numpy(im_dets_obsv).float()
        item['target_lbl'] = target_lbl
        item['path_length'] = path_len
        return item


    def __len__(self):
        return len(self.images)


    def get_step_data(self, next_ims, scenes, scales):    
        # Given the next image in the sequence retrieve the relevant info
        # Assumes that next_im is not a collision
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
            scene_seg = self.sseg_data[scene.encode()]
            scene_dets = self.detection_data[scene]

            imgData, points2D, local3D = dh.getImageData(self.datasetPath, next_im, scene, 
                                                            self.cropSize, self.orig_res, self.pixFormat, self.normalize)
            im_sseg = dh.get_sseg(next_im, scene_seg, self.cropSize)
            im_dets = dh.get_det_mask(next_im, scene_dets, self.cropSize, self.dets_nClasses, self.labels_to_index)
            im_obsv = dh.getImageData(self.datasetPath, next_im, scene, self.cropSizeObsv, self.orig_res, 
                                                                            self.pixFormat, self.normalize, get3d=False)
            im_dets_obsv = dh.get_det_mask(next_im, scene_dets, self.cropSizeObsv, 1, self.labels_to_index)

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




class AVD(Dataset):
    # AVD class that pre-samples episodes for MapNet training
    def __init__(self, par, seq_len, nEpisodes, scene_list, action_list, with_shortest_path=False):
        self.datasetPath = par.avd_root
        self.cropSize = par.crop_size
        self.orig_res = par.orig_res
        self.normalize = True
        self.pixFormat = "NCHW"
        self.scene_list = scene_list
        self.n_episodes = nEpisodes # how many episodes to sample per scene
        self.seq_len = seq_len # this is used when training MapNet with constant sequence length
        self.actions = action_list
        self.dets_nClasses = par.dets_nClasses
        # Read the semantic segmentations
        self.sseg_data = np.load(par.sseg_file_path, encoding='bytes', allow_pickle=True).item()
        # Load the detections, the object masks are created during getitem()
        self.detection_data, self.labels_to_cats, self.labels_to_index = dh.load_detections(par, self.scene_list)
        # Pre-sample the episodes and organize in dictionaries when the dataset instance is created.
        if with_shortest_path:
            self.sample_episodes()
        else:
            self.sample_episodes_random()

    
    def sample_episodes(self):
        # This function chooses the actions through shortest path
        epi_id=0 # episode id
        im_paths, pose, scene_name, scene_scale = {}, {}, {}, {}
        for scene in self.scene_list:
            annotations, scale, im_names_all, world_poses, directions = dh.load_scene_info(self.datasetPath, scene)
            # Create the graph of the environment
            graph = dh.create_scene_graph(annotations, im_names=im_names_all, action_set=self.actions)
            scene_epi_count = 0
            while scene_epi_count < self.n_episodes:
                # Randomly select two nodes and sample a trajectory across their shortest path
                idx = np.random.randint(len(im_names_all), size=2)
                im_name_0 = im_names_all[idx[0]]
                im_name_1 = im_names_all[idx[1]]
                # organize the episodes into dictionaries holding different information
                if nx.has_path(graph, im_name_0, im_name_1):
                    path = nx.shortest_path(graph, im_name_0, im_name_1) # sequence of nodes leading to goal
                    if len(path) >= self.seq_len:
                        poses_epi = []
                        for i in range(self.seq_len):
                            next_im = path[i]
                            im_idx = np.where(im_names_all==next_im)[0]
                            pos_tmp = world_poses[im_idx][0] * scale # 3 x 1
                            pose_x_gt = pos_tmp[0,:]
                            pose_z_gt = pos_tmp[2,:]
                            dir_tmp = directions[im_idx][0] # 3 x 1
                            dir_gt = np.arctan2(dir_tmp[2,:], dir_tmp[0,:])[0] # [-pi,pi], assumes that the 0 direction is to the right
                            poses_epi.append([pose_x_gt, pose_z_gt, dir_gt])

                        im_paths[epi_id] = np.asarray(path[:self.seq_len])
                        pose[epi_id] = np.asarray(poses_epi, dtype=np.float32)
                        scene_name[epi_id] = scene
                        scene_scale[epi_id] = scale
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
        for scene in self.scene_list:
            annotations, scale, im_names_all, world_poses, directions = dh.load_scene_info(self.datasetPath, scene)           
            # Create the graph of the environment
            graph = dh.create_scene_graph(annotations, im_names=im_names_all, action_set=self.actions)
            scene_epi_count = 0
            while scene_epi_count < self.n_episodes:
                # Randomly select an image index as the starting position
                idx = np.random.randint(len(im_names_all), size=1)
                im_name_0 = im_names_all[idx[0]]          
                # organize the episodes into dictionaries holding different information
                poses_epi, path = [], []
                for i in range(self.seq_len):
                    if i==0:
                        current_im = im_name_0
                    else:
                        # randomly choose the action
                        sel_action = self.actions[np.random.randint(len(self.actions), size=1)[0]]
                        next_im = annotations[current_im][sel_action]
                        if not(next_im==''):
                            current_im = next_im
                    path.append(current_im)
                    im_idx = np.where(im_names_all==current_im)[0]
                    pos_tmp = world_poses[im_idx][0] * scale # 3 x 1
                    pose_x_gt = pos_tmp[0,:]
                    pose_z_gt = pos_tmp[2,:]
                    dir_tmp = directions[im_idx][0] # 3 x 1
                    dir_gt = np.arctan2(dir_tmp[2,:], dir_tmp[0,:])[0] # [-pi,pi]
                    poses_epi.append([pose_x_gt, pose_z_gt, dir_gt])

                im_paths[epi_id] = np.asarray(path)
                pose[epi_id] = np.asarray(poses_epi, dtype=np.float32)
                scene_name[epi_id] = scene
                scene_scale[epi_id] = scale
                epi_id+=1
                scene_epi_count+=1
                        
        self.im_paths = im_paths
        self.pose = pose
        self.scene_name = scene_name
        self.scene_scale = scene_scale


    def __len__(self):
        return len(self.im_paths)

    # Returns the complete index episode
    def __getitem__(self, index):
        item = {}
        path = self.im_paths[index]
        poses_epi = self.pose[index]
        scene = self.scene_name[index]
        scale = self.scene_scale[index]
        scene_seg = self.sseg_data[scene.encode()] # convert string to byte
        scene_dets = self.detection_data[scene]

        imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
        ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
        dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
        points2D, local3D = [], []
        for i in range(len(path)): # seq_len
            im_name = path[i]
            imgData, points2D_step, local3D_step = dh.getImageData(self.datasetPath, im_name, 
                                                        scene, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
            imgs[i,:,:,:] = imgData
            points2D.append(points2D_step) # points2D and local3D for each step have different sizes, so save them as lists of lists 
            local3D.append(local3D_step)
            # Get the semantic segmentations and detection masks
            ssegs[i,:,:,:] = dh.get_sseg(im_name, scene_seg, self.cropSize)
            dets[i,:,:,:] = dh.get_det_mask(im_name, scene_dets, self.cropSize, self.dets_nClasses, self.labels_to_index)

        # Need to get the relative poses (towards the first frame) for the ground-truth
        rel_poses = dh.relative_poses(poses=poses_epi)

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



class AVD_IL(Dataset):
    # AVD class that presamples episodes and the costs of actions which are used for supervision
    # Tailored to work for imitation learning
    def __init__(self, par, seq_len, nEpisodes, scene_list, action_list):
        self.datasetPath = par.avd_root
        self.cropSize = par.crop_size
        self.cropSizeObsv = par.crop_size_obsv
        self.orig_res = par.orig_res
        self.normalize = True
        self.pixFormat = "NCHW"
        self.scene_list = scene_list
        self.n_episodes = nEpisodes # how many episodes to sample per scene
        self.seq_len = seq_len
        self.actions = action_list
        self.dets_nClasses = par.dets_nClasses
        # Read the semantic segmentations
        self.sseg_data = np.load(par.sseg_file_path, encoding='bytes', allow_pickle=True).item()
        # Load the detections, the object masks are created during getitem
        self.detection_data, self.labels_to_cats, self.labels_to_index = dh.load_detections(par, self.scene_list)
        self.cat_dict = par.cat_dict # target category names to labels dictionary
        # Need to collect the image names for the goals
        self.targets_data = np.load(par.targets_file_path, encoding='bytes', allow_pickle=True).item()
        # Need to pre-calculate the graphs of the scenes for every target
        self.graphs_dict = dh.get_scene_target_graphs(self.datasetPath, self.cat_dict, self.targets_data, self.actions)
        self.sample_episodes()


    def sample_episodes(self):
        # Each episode should contain:
        # List of images, list of actions, cost of every action, scene, scale, collision indicators
        epi_id=0 # episode id
        im_paths, action_paths, cost_paths, scene_name, scene_scale, target_lbls, pose_paths, collisions = {}, {}, {}, {}, {}, {}, {}, {}
        for scene in self.scene_list:
            annotations, scale, im_names_all, world_poses, directions = dh.load_scene_info(self.datasetPath, scene)
            scene_epi_count = 0
            while scene_epi_count < self.n_episodes:
                # Randomly select an image index as the starting position      
                idx = np.random.randint(len(im_names_all), size=1)
                im_name_0 = im_names_all[idx[0]]
                # Randomly select a target that exists in that scene
                candidates = dh.candidate_targets(scene, self.cat_dict, self.targets_data) # Get the list of possible targets
                idx_cat = np.random.randint(len(candidates), size=1)
                cat = candidates[idx_cat[0]]
                target_lbl = self.cat_dict[cat]
                graph = self.graphs_dict[target_lbl][scene] # to be used to get the ground-truth
                # Choose whether the episode's observations are going to be decided by the
                # teacher (best action) or randomly
                choice = np.random.randint(2, size=1)[0] # if 1 then do teacher
                im_seq, action_seq, cost_seq, poses_seq, collision_seq = [], [], [], [], []
                im_seq.append(im_name_0)
                current_im = im_name_0
                # get the ground-truth cost for each next state
                cost_seq.append(dh.get_state_action_cost(current_im, self.actions, annotations, graph))
                poses_seq.append(dh.get_im_pose(im_names_all, current_im, world_poses, directions, scale))
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
                    next_im = annotations[current_im][sel_action]
                    if not(next_im==''): # if there is a collision then keep the same image
                        current_im = next_im
                        collision_seq.append(0)
                    else:
                        collision_seq.append(1)
                    im_seq.append(current_im)
                    action_seq.append(sel_action)
                    # get the ground-truth pose
                    poses_seq.append(dh.get_im_pose(im_names_all, current_im, world_poses, directions, scale))
                    cost_seq.append(dh.get_state_action_cost(current_im, self.actions, annotations, graph))

                im_paths[epi_id] = np.asarray(im_seq)
                action_paths[epi_id] = np.asarray(action_seq)
                cost_paths[epi_id] = np.asarray(cost_seq, dtype=np.float32)
                scene_name[epi_id] = scene
                scene_scale[epi_id] = scale
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

    # Returns the index episode
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
        scene_seg = self.sseg_data[scene.encode()] # convert string to byte
        scene_dets = self.detection_data[scene]

        imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
        imgs_obsv = np.zeros((self.seq_len, 3, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
        ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
        dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
        dets_obsv = np.zeros((self.seq_len, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
        points2D, local3D = [], [] #{}, {}
        for i in range(len(im_path)): # seq_len
            im_name = im_path[i]
            #print(index, i)
            imgData, points2D_step, local3D_step = dh.getImageData(self.datasetPath, im_name, 
                                                        scene, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
            imgs[i,:,:,:] = imgData
            points2D.append(points2D_step) # points2D and local3D for each step have different sizes, so save them as lists of lists 
            local3D.append(local3D_step)
            imgs_obsv[i,:,:,:] = dh.getImageData(self.datasetPath, im_name, scene, self.cropSizeObsv, 
                                                    self.orig_res, self.pixFormat, self.normalize, get3d=False)
            # Load the semantic segmentations, detection masks and egocentric detection observation
            ssegs[i,:,:,:] = dh.get_sseg(im_name, scene_seg, self.cropSize)
            dets[i,:,:,:] = dh.get_det_mask(im_name, scene_dets, self.cropSize, self.dets_nClasses, self.labels_to_index)
            dets_obsv[i,:,:,:] = dh.get_det_mask(im_name, scene_dets, self.cropSizeObsv, 1, self.labels_to_index)

        # Need to get the relative poses (towards the first frame) for the ground-truth
        rel_poses = dh.relative_poses(poses=abs_poses)

        item["images"] = torch.from_numpy(imgs).float()
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
        item['dets'] = torch.from_numpy(dets).float()
        item['images_obsv'] = torch.from_numpy(imgs_obsv).float()
        item['dets_obsv'] = torch.from_numpy(dets_obsv).float()
        return item

























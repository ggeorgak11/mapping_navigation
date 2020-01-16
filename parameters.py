
import os
import numpy as np


class Parameters(object):
    # Common parameter values between mapnet and navigation modules
    def __init__(self):
        # Data paths
        self.avd_root = '../ActiveVisionDataset/'
        self.sseg_file_path = self.avd_root + "Meta/sseg_crf.npy"
        self.targets_file_path = self.avd_root + "Meta/annotated_targets.npy"
        self.det_dir_path = self.avd_root + "Meta/Detections/"
        self.label_map_path = self.avd_root + "Meta/coco_labels_reduced.txt"
        self.label_index_path = self.avd_root + "Meta/coco_labels_to_index.txt"
        # Path to code
        self.src_root = '../src/'
        # AVD images dimensions
        self.orig_res = (1920,1080)
        # Input images dimensions (resized to)
        self.crop_size = (160,90) #(320,180)
        self.train_scene_list = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2',
                                    'Home_004_1', 'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1','Home_010_1']
        # Set of actions from the avd annotations
        self.action_list = ['rotate_ccw', 'rotate_cw', 'forward']
        # Iteration intervals during training
        self.save_interval = 500
        self.show_interval = 1
        self.plot_interval = 100
        self.test_interval = 500


class ParametersMapNet(Parameters):
    # Specific parameters for the MapNet module
    def __init__(self):
        Parameters.__init__(self)
        ## Choose the model id, this applies during testing as well
        self.model_id = "00"
        self.model_dir = self.src_root + "output/" + self.model_id + "/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        ## Map parameters
        # Dimension of observation grid
        self.observation_dim = (21,21)
        # Dimension of global map 
        self.global_map_dim = (29,29)
        # Dimension (xs, zs) of each bin in mm
        self.cell_size = 300
        # Embeddings sizes for img, sseg, det grids
        self.img_embedding = 32
        self.sseg_embedding = 16
        self.dets_embedding = 16
        # Number of sseg classes, following the NYUv2 40 labels 
        self.sseg_labels = 40
        # Number of det classes, reduced COCO classes 
        self.dets_nClasses = 40 
        # Enable/disable, img, sseg, det grids 
        self.with_img = False
        self.with_sseg = True
        self.with_dets = True
        # Enable/disable embedding or raw sseg, dets
        self.use_raw_sseg = False
        self.use_raw_dets = False
        # How each cell embedding is updated in the global map: 'lstm', 'fc', 'avg' 
        self.update_type = 'lstm'
        # Embedding dimensions based on previous choices
        if self.use_raw_sseg:
            self.sseg_embedding = self.sseg_labels
        if self.use_raw_dets:
            self.dets_embedding = self.dets_nClasses
        self.map_embedding = self.with_img*self.img_embedding + self.with_sseg*self.sseg_embedding + self.with_dets*self.dets_embedding
        self.input_flags = (self.with_img, self.with_sseg, self.with_dets, self.use_raw_sseg, self.use_raw_dets)
        # Number of orientations used in the rotation stack
        self.orientations = 12
        # Padding for cross-correlation (and deconvolution) to get the right output dim
        self.pad = int((self.observation_dim[1]-1)/2.0)
        # Sequence length which mapNet is run. Used 5, 20. This applies during testing as well. 
        self.seq_len = 5

        ## Training params
        # Number of sampled episodes per scene
        self.epi_per_scene = 3000
        self.batch_size = 12 
        # When sampling the episodes use shortest path or not (see dataloader.AVD)
        self.with_shortest_path = False
        # Binary cross entropy "BCE" and cross-entropy "CEL" losses are supported
        self.loss_type = "CEL"
        self.nEpochs = 10
        self.lr_rate = 1e-5
        # After how many epochs to reduce the learning rate
        self.step_size = 3
        # Decaying factor at every step size
        self.gamma = 0.5 

        ## Evaluation params
        self.test_scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1'] # example scenes
        # Apply model trained at test_iter iterations
        self.test_iter = 2500


class Parameters_IL(Parameters):
    # Parameters for training the Navigation model with Imitation Learning
    def __init__(self):
        Parameters.__init__(self)
        ## Choose the model id, this applies during testing as well
        self.model_id = "IL_00" #"IL_2" 
        self.model_dir = self.src_root + "output/" + self.model_id + "/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        ## Choose the trained mapNet model to deploy during navigation model training
        ## Note - The following params in ParametersMapNet class should be set accordingly to the chosen mapnet here:
        ## : grid and embedding dimensions, enabled img, sseg, det grids, update_type, orientations 
        self.mapNet_model_id = "00" #"11" #"8" #"3" # which mapNet model to deploy in QNet training
        self.mapNet_model_dir = self.src_root + "output/" + self.mapNet_model_id + "/"
        self.mapNet_iters = 1000
        ## Parameters for mapnet
        # Enable/disable using ground-truth instead of predicted pose when running mapNet
        self.use_p_gt = False
        # Enable/disable backprop navigation gradients to the mapNet model while training
        self.finetune_mapNet = True

        ## Parameters for ILNet
        # Enable/disable using egocentric observations as input
        self.use_ego_obsv = True
        # Dimension of egocentric observations (resized to)
        self.crop_size_obsv = (224,224)
        # Number of conv channels when extracting the embedding from map and pose prediction
        self.conv_embedding = 8
        # FC dimension for all input embeddings
        self.fc_dim = 128
        # Categories and labels used in the experiments
        self.cat_dict = {'dining_table':0, 'fridge':1, 'tv':2, 'couch':3, 'microwave':4}
        self.lbl_to_cat = {1:'fridge', 3:'couch', 2:'tv', 0:'dining_table', 4:'microwave'}
        self.nTargets = len(self.cat_dict)
        # Number of det classes, reduced COCO classes 
        self.dets_nClasses = 40

        ## Training params
        self.batch_size = 8
        # Number of sampled episodes per scene
        self.epi_per_scene = 5000
        # Sequence length during training
        self.seq_len = 10
        self.nEpochs = 10
        self.lr_rate = 1e-3
        # Navigation loss weight
        self.loss_weight = 10
        # After how many epochs to reduce the learning rate
        self.step_size = 3
        # Decaying factor at every step size
        self.gamma = 0.5 
        # Params on how to select the minibatch (see train_NavNet.select_minibatch)
        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 1000      

        ## Evaluation params
        # Enable/disable the test set used in our experiments
        self.use_predefined_test_set = True
        self.predefined_confs_file = self.avd_root + "Meta/init_configs.npy"
        self.predefined_test_scenes = ['Home_011_1', 'Home_013_1', 'Home_016_1']
        # Maximum length of a sampled episode during testing (when predefined test set is not used)
        self.max_shortest_path = 20 
        # Max steps to end an episode
        self.max_steps = 100
        # Enable/disable failing an episode when a collision happens
        self.stop_on_collision = False
        # How many steps away from goal to declare success, following other work compared in the experiments
        self.steps_from_goal = 5
        # Apply model trained at test_iter iterations
        self.test_iters = 2500



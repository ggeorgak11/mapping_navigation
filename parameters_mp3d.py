import os
import numpy as np


class ParametersMapNet_MP3D(object):
    def __init__(self):
        self.mp3d_root = '/scratch/yli44/Datasets/MP3D'

        # scene_level
        self.train_scene_list = ['7y3sRwLe3Va_1', 
        '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1', 'e9zR4mvMWw7_0',
        'GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1',
        'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0', 'V2XKFyX4ASd_1', 'V2XKFyX4ASd_2',
        'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0',
        '2t7WUuJeko7_0', 'RPmz2sHmrrY_0', 'WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0'
        ]
        self.test_scene_list = ['2t7WUuJeko7_0', 'RPmz2sHmrrY_0', 'WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0']

        self.action_list = ['rotate_ccw', 'rotate_cw', 'forward'] # need this to create the graph

        self.orig_res = (256, 256)
        self.crop_size = (64, 64) #(320,180)
        self.epi_per_scene = 4000 #2000 # number of sampled episodes per scene
        self.batch_size = 24 #16 #4
        self.seq_len = 20
        self.with_shortest_path = False #True

        #=====================================================================================================
        #'''
        self.src_root = '/scratch/yli44/pyMapNet_mp3d/'
        self.model_id = '12'
        self.model_dir = self.src_root + "output/" + self.model_id + "/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.custom_resnet = False

        self.observation_dim = (21,21)
        self.global_map_dim = (29,29)
        self.cell_size = 0.3 # mm  #0.3 # meters, bin dimensions xs, zs
        self.img_embedding = 32
        self.sseg_embedding = 16
        self.dets_embedding = 16
        self.sseg_labels = 40 #300 # Following the NYUv2 40 labels
        self.dets_nClasses = 40 #91 # COCO 90 + unknown
        #####
        self.with_img = True
        self.with_sseg = True
        self.with_dets = True
        self.use_raw_sseg = False
        self.use_raw_dets = False
        self.update_type = 'lstm' # 'lstm', 'fc', 'avg'
        # change the embedding dimensions if the inputs are passed raw
        if self.use_raw_sseg:
            self.sseg_embedding = self.sseg_labels
        if self.use_raw_dets:
            self.dets_embedding = self.dets_nClasses
        self.map_embedding = self.with_img*self.img_embedding + self.with_sseg*self.sseg_embedding + self.with_dets*self.dets_embedding #32
        self.input_flags = (self.with_img, self.with_sseg, self.with_dets, self.use_raw_sseg, self.use_raw_dets)

        self.orientations = 12
        self.pad = int((self.observation_dim[1]-1)/2.0) # padding for cross-correlation (and deconvolution) to get the right output dim

        # training params
        self.loss_type = "NLL" #"BCE" #"NLL" # only binary cross entropy and negative log likelihood (cross-entropy) are supported 
        self.nEpochs = 100
        self.lr_rate = 1e-5 #1e-4 #1e-5
        #self.momentum = 0.9
        #self.weight_decay = 0.0005
        self.step_size = 3 #250 #250 # 500 # after how many epochs to reduce the learning rate
        self.gamma = 0.5 #0.1 # decaying factor at every step size

        self.save_interval = 500
        self.show_interval = 5
        self.plot_interval = 100
        self.test_interval = 500

        # Evaluation params
        self.test_batch_size = 1
        #'''

class ParametersIL_MP3D(object):
    def __init__(self):
        self.mp3d_root = '/scratch/yli44/Datasets/MP3D'

        # scene_level
        self.train_scene_list = ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1',
        'e9zR4mvMWw7_0','GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1',
        'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0', 'V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0']
        self.test_scene_list = ['2t7WUuJeko7_0', 'RPmz2sHmrrY_0', 'WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0']

        self.action_list = ['rotate_ccw', 'rotate_cw', 'forward'] # need this to create the graph

        self.orig_res = (256, 256)
        self.crop_size = (64, 64) #(320,180)
        self.crop_size_obsv = (224, 224)

        self.sseg_labels = 40 #300 
        self.dets_nClasses = 40

        self.mapNet_src_root = '/scratch/yli44/pyMapNet_mp3d/'
        self.mapNet_model_id = "12" #"8" #"3" # which mapNet model to deploy in QNet training
        self.mapNet_model_dir = self.mapNet_src_root + "output/" + self.mapNet_model_id + "/"
        self.mapNet_iters = 1000 #6000 #3000 #12000
        self.use_p_gt = False #True # use ground-truth instead of predicted pose when running mapNet
        self.finetune_mapNet = True # backprop navigation gradients to the mapNet model while training


        self.model_id = "IL_9" #"IL_2" # ** Try the IL_1 (testing) with different mapNets, even though it was not trained with them
        self.model_dir = self.mapNet_src_root + "output/" + self.model_id + "/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.test_iters = 1500 #5500 #5000 #2500

        # params for ILNet
        self.use_ego_obsv = True #False
        self.conv_embedding = 8
        self.fc_dim = 128 # fc dimensions for all embeddings

        self.cat_dict = {'dining_table':0, 'fridge':1, 'tv':2, 'couch':3, 'microwave':4}
        self.lbl_to_cat = {1:'fridge', 3:'couch', 2:'tv', 0:'dining_table', 4:'microwave'}
        self.nTargets = len(self.cat_dict)

        self.batch_size = 24 #24 #32
        self.epi_per_scene = 4000 #5000 #1000 #2000 # number of sampled episodes per scene
        self.seq_len = 10 # sequence length for training

        self.max_shortest_path = 20 # maximum length of a sampled episode during testing (when arsalan's test ids are not used)
        self.max_steps = 100 #50 # during testing
        self.stop_on_collision = False # used during testing
        self.steps_from_goal = 5 # how many steps away from goal to declare success, used during evaluation

        self.nEpochs = 1001
        self.lr_rate = 1e-3
        self.loss_weight = 10
        self.step_size = 2 #250 #250 # 500 # after how many epochs to reduce the learning rate
        self.gamma = 0.5 #0.1 # decaying factor at every step size

        # how to select the minibatch params
        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 1000 #3000 #5000 # 2000

        self.save_interval = 500
        self.show_interval = 1
        self.plot_interval = 100
        self.test_interval = 500

class ParametersQNet_MP3D(object):
    def __init__(self):
        self.mp3d_root = '/scratch/yli44/Datasets/MP3D'
        self.orig_res = (256, 256)
        self.crop_size = (64, 64) #(320,180)
        self.crop_size_obsv = (224, 224)

        #self.train_scene_list = ['Home_001_1']
        self.train_scene_list = ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1',
        'e9zR4mvMWw7_0','GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1',
        'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0', 'V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 
        'zsNo4HB9uLZ_0']
        self.test_scene_list = ['2t7WUuJeko7_0', 'RPmz2sHmrrY_0', 'WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0']

        self.start_pos_per_scene = 1000 # how many starting positions to sample per scene
        self.max_shortest_path = 20 #30 #5 # maximum length of a sampled episode
        self.action_list = ['rotate_ccw', 'rotate_cw', 'forward']

        # target categories and their corresponding labels in the NYU 40 class list 
        #self.cat_dict = {'fridge':23, 'couch':5, 'tv':24, 'dining_table':6} # microwave is not in the NYU 40 class list
        self.cat_dict = {'dining_table':0, 'fridge':1, 'tv':2, 'couch':3, 'microwave':4}
        #self.lbl_to_cat = {23:'fridge', 5:'couch', 24:'tv', 6:'dining_table'}
        self.lbl_to_cat = {1:'fridge', 3:'couch', 2:'tv', 0:'dining_table', 4:'microwave'}
        self.nTargets = len(self.cat_dict)
        self.dets_nClasses = 40 #91

        self.stop_on_collision = False #True

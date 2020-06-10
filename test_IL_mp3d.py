import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import math
import random
from dataloader_mp3d import MP3D_IL, MP3D_online
from mapNet import MapNet
from IL_Net import Encoder
from parameters_mp3d import ParametersIL_MP3D, ParametersMapNet_MP3D, ParametersQNet_MP3D
import helper as hl
import data_helper as dh
import data_helper_mp3d as dhm
import networkx as nx
import pickle
#import vis_test as vis

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def softmax(x):
	scoreMatExp = np.exp(np.asarray(x))
	return scoreMatExp / scoreMatExp.sum(0)

#def check_if_stuck(action_seq):
    # if the agent is lost it tends to predict the same rotation over and over
    # if this happens 5 consecutive times, then move forward
#    a = []


def prepare_mapNet_input(ex):
    img = ex["image"]
    points2D = ex["points2D"]
    local3D = ex["local3D"]
    sseg = ex["sseg"]
    dets = ex['dets']
    # for now assume that test_batch_size=1
    imgs_batch = img.unsqueeze(0)
    sseg_batch = sseg.unsqueeze(0)
    dets_batch = dets.unsqueeze(0)
    points2D_batch, local3D_batch = [], [] # add another dimension for the batch
    points2D_batch.append(points2D)
    local3D_batch.append(local3D)
    return (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), dets_batch.cuda())


def evaluate_ILNet(parIL, parMapNet, mapNet, ego_encoder, test_iter, test_ids, test_data, action_list, test=False):
    print("\nRunning validation on ILNet!")
    with torch.no_grad():
        policy_net = hl.load_model(model_dir=parIL.model_dir, model_name="ILNet", test_iter=test_iter)
        acc, epi_length = 0, 0
        episode_results = {} # store predictions in order to visualize
        for i in test_ids:
            test_ex = test_data[i]
            # Get all info for the starting position
            mapNet_input_start = prepare_mapNet_input(ex=test_ex)
            target_lbl = test_ex["target_lbl"]
            im_obsv = test_ex['image_obsv'].cuda()
            dets_obsv = test_ex['dets_obsv'].cuda()
            tvec = torch.zeros(1, parIL.nTargets).float().cuda()
            tvec[0,target_lbl] = 1
            # We need to keep other info to allow us to do the steps later
            image_name, scene, scale = [], [], []
            image_name.append(test_ex['image_name'])
            scene.append(test_ex['scene'])
            scale.append(test_ex['scale'])

            # get the ground-truth pose, which is the relative pose with respect to the first image
            # we do not need to do it for the first image, since we have p0 in mapNet
            # we need to store the poses so that we can estimate the gt relative pose during the episode
            if parIL.use_p_gt:
                scene_folder = '{}/{}'.format(self.datasetPath, scene)
                info = dhm.load_scene_info(scene_folder)
                im_names_all = list(info.keys()) # info 0 # list of image names in the scene
                im_names_all = np.hstack(im_names_all) # flatten the array
                start_abs_pose = dhm.get_image_poses(info, im_names_all, image_name, scale[0])

            # Get state from mapNet
            p_, map_ = mapNet.forward_single_step(local_info=mapNet_input_start, t=0, 
                                                    input_flags=parMapNet.input_flags, update_type=parMapNet.update_type)
            collision_ = torch.tensor([0], dtype=torch.float32).cuda() # collision indicator is 0
            if parIL.use_ego_obsv:
                enc_in = torch.cat((im_obsv, dets_obsv), 0).unsqueeze(0)
                ego_obsv_feat = ego_encoder(enc_in) # 1 x 512 x 1 x 1
                state = (map_, p_, tvec, collision_, ego_obsv_feat)
            else:
                state = (map_, p_, tvec, collision_) 
            current_im = image_name[0] #.copy()

            done=0
            image_seq, action_seq = [], []
            image_seq.append(current_im)
            policy_net.hidden = policy_net.init_hidden(batch_size=1, state_items=len(state)-1)
            deterministic = False
            for t in range(1, parIL.max_steps+1):
                pred_costs = policy_net(state, parIL.use_ego_obsv) # apply policy for single step
                pred_costs = pred_costs.view(-1).cpu().numpy()
                if deterministic:
                    pred_label = np.argmin(pred_costs)
                    pred_action = action_list[pred_label]
                else:
                    # choose the action with a certain prob
                    pred_probs = softmax(-pred_costs)
                    pred_label = np.random.choice(len(action_list), 1, p=pred_probs)[0]
                    pred_action = action_list[pred_label]

                # get the next image, check collision and goal
                next_im = test_data.scene_infos[scene[0]][current_im][pred_action]
                image_seq.append(next_im)
                action_seq.append(pred_action)
                print(t, current_im, pred_action, next_im)
                if not(next_im==''): # not collision case
                    #print("Not collision!")
                    collision = 0
                    # check for goal
                    path_dist = len(nx.shortest_path(test_data.graphs_dict[target_lbl][scene[0]], next_im, "goal")) - 2
                    if path_dist <= parIL.steps_from_goal: # GOAL!
                        acc += 1
                        epi_length += t
                        done=1
                        break
                    # get next state from mapNet
                    batch_next, obsv_batch_next = test_data.get_step_data(next_ims=[next_im], scenes=scene, scales=scale)
                    if parIL.use_p_gt:
                        next_im_abs_pose = dhm.get_image_poses(info, im_names_all, [next_im], scale[0])
                        abs_poses = np.concatenate((start_abs_pose, next_im_abs_pose), axis=0)
                        rel_poses = dhm.relative_poses(poses=abs_poses)
                        next_im_rel_pose = np.expand_dims(rel_poses[1,:], axis=0)
                        p_gt = dh.build_p_gt(parMapNet, pose_gt_batch=np.expand_dims(next_im_rel_pose, axis=1)).squeeze(1)
                        p_next, map_next = mapNet.forward_single_step(local_info=batch_next, t=t, input_flags=parMapNet.input_flags,
                                                                map_previous=state[0], p_given=p_gt, update_type=parMapNet.update_type)
                    else:
                        p_next, map_next = mapNet.forward_single_step(local_info=batch_next, t=t, 
                                            input_flags=parMapNet.input_flags, map_previous=state[0], update_type=parMapNet.update_type)
                    if parIL.use_ego_obsv:
                        enc_in = torch.cat(obsv_batch_next, 1)
                        ego_obsv_feat = ego_encoder(enc_in) # b x 512 x 1 x 1
                        state = (map_next, p_next, tvec, torch.tensor([collision], dtype=torch.float32).cuda(), ego_obsv_feat)
                    else:
                        state = (map_next, p_next, tvec, torch.tensor([collision], dtype=torch.float32).cuda())
                    current_im = next_im

                else: # collision case
                    #print("Collision!")
                    collision = 1
                    if parIL.stop_on_collision:
                        break
                    if parIL.use_ego_obsv:
                        state = (state[0], state[1], state[2], torch.tensor([collision], dtype=torch.float32).cuda(), state[4])
                    else:
                        state = (state[0], state[1], state[2], torch.tensor([collision], dtype=torch.float32).cuda())
                
            episode_results[i] = (image_seq, action_seq, target_lbl, done) 

        #vis.vis_image_seq(par=parIL, episode_results=episode_results, scene=scene[0])
        #vis.visualize_episode_res(par=parIL, episode_results=episode_results, scene=scene[0])
        #raise Exception("!!!")

        episode_results_path = parIL.model_dir+'episode_results_'+str(test_iter)+'.pkl'
        if test:
            episode_results_path = parIL.model_dir+'episode_results_test_'+str(test_iter)+'.pkl'
        with open(episode_results_path, 'wb') as f:
            pickle.dump(episode_results, f)
        
        res_file = open(parIL.model_dir+"val_"+parIL.model_id+".txt", "a+")
        success_rate = acc / float(len(test_ids))
        if acc > 0:
            mean_epi_length = epi_length / float(acc) #float(len(test_ids))
        else:
            mean_epi_length = 0
        print("Test iter:", test_iter, "Success rate:", success_rate, "Mean epi length:", mean_epi_length)
        res_file.write("Test iter:" + str(test_iter) + "\n")
        res_file.write("Test set:" + str(len(test_ids)) + "\n")
        res_file.write("Success rate:" + str(success_rate) + "\n")
        res_file.write("Mean episode length:" + str(mean_epi_length) + "\n")
        res_file.write("\n")
        res_file.close()



if __name__ == '__main__':
    parQNet = ParametersQNet_MP3D()
    parMapNet = ParametersMapNet_MP3D()
    parIL = ParametersIL_MP3D()
    action_list = np.asarray(parMapNet.action_list)
    use_arsalan_test_set = True

    '''
    if use_arsalan_test_set:
        # Open Arsalan's initial configurations and load them in avd
        init_confs_file = parIL.avd_root + "AVD_Minimal/Meta/all_init_configs.npy"
        init_confs = np.load(init_confs_file, encoding='bytes', allow_pickle=True).item()
        arsalan_test_scenes = ['Home_011_1', 'Home_013_1', 'Home_016_1']
        avd = AVD_online(par=parQNet, nStartPos=0, scene_list=arsalan_test_scenes, 
                                                action_list=action_list, init_configs=init_confs)
    '''
    #else:
    # sample starting positions and targets from the mp3d_online class
    init_confs = np.load('{}/all_init_configs.npy'.format(parQNet.mp3d_root), allow_pickle=True).item()
    #init_confs = np.load('{}/small_init_configs.npy'.format(parQNet.mp3d_root), allow_pickle=True).item()
    mp3d = MP3D_online(par=parIL, nStartPos=10, scene_list=parIL.test_scene_list, action_list=action_list, 
        init_configs=init_confs)

    test_ids = list(range(len(mp3d)))
    #print(len(test_ids))
    #test_ids = [186]
    #raise Exception("111")

    # Need to load the trained MapNet
    if parIL.finetune_mapNet: # choose whether to use a finetuned mapNet model or not
        mapNet_model = hl.load_model(model_dir=parIL.model_dir, model_name="MapNet", test_iter=parIL.test_iters)
    else:
        mapNet_model = hl.load_model(model_dir=parIL.mapNet_model_dir, model_name="MapNet", test_iter=parIL.mapNet_iters)
    # If we are not using a trained mapNet model then define a new one
    #mapNet_model = MapNet(parMapNet, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags) #Encoder(par)
    #mapNet_model.cuda()
    #mapNet_model.eval()
    
    ego_encoder = Encoder()
    ego_encoder.cuda()
    ego_encoder.eval()


    evaluate_ILNet(parIL, parMapNet, mapNet_model, ego_encoder, test_iter=parIL.test_iters, 
                        test_ids=test_ids, test_data=mp3d, action_list=action_list, test=use_arsalan_test_set) 
                     
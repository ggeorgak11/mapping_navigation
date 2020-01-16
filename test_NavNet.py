import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
from dataloader import AVD_online
from mapNet import MapNet
from IL_Net import Encoder
from parameters import ParametersMapNet, Parameters_IL
import helper as hl
import data_helper as dh
import networkx as nx
import pickle


def softmax(x):
	scoreMatExp = np.exp(np.asarray(x))
	return scoreMatExp / scoreMatExp.sum(0)


def prepare_mapNet_input(ex):
    img = ex["image"]
    points2D = ex["points2D"]
    local3D = ex["local3D"]
    sseg = ex["sseg"]
    dets = ex['dets']
    # test_batch_size=1
    imgs_batch = img.unsqueeze(0)
    sseg_batch = sseg.unsqueeze(0)
    dets_batch = dets.unsqueeze(0)
    points2D_batch, local3D_batch = [], [] # add another dimension for the batch
    points2D_batch.append(points2D)
    local3D_batch.append(local3D)
    return (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), dets_batch.cuda())


def evaluate_NavNet(parIL, parMapNet, mapNet, ego_encoder, test_iter, test_ids, test_data, action_list):
    print("\nRunning validation on NavNet!")
    with torch.no_grad():
        policy_net = hl.load_model(model_dir=parIL.model_dir, model_name="ILNet", test_iter=test_iter)
        acc, epi_length, path_ratio = 0, 0, 0
        episode_results = {} # store predictions
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
            shortest_path_length = test_ex['path_length']

            if parIL.use_p_gt:
                # get the ground-truth pose, which is the relative pose with respect to the first image
                info, annotations, _ = dh.load_scene_info(parIL.avd_root, scene[0])
                im_names_all = info['image_name'] # info 0 # list of image names in the scene
                im_names_all = np.hstack(im_names_all) # flatten the array
                start_abs_pose = dh.get_image_poses(info, im_names_all, image_name, scale[0]) # init pose of the episode # 1 x 3  

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
            current_im = image_name[0]

            done=0
            image_seq, action_seq = [], []
            image_seq.append(current_im)
            policy_net.hidden = policy_net.init_hidden(batch_size=1, state_items=len(state)-1)
            for t in range(1, parIL.max_steps+1):
                pred_costs = policy_net(state, parIL.use_ego_obsv) # apply policy for single step
                pred_costs = pred_costs.view(-1).cpu().numpy()
                # choose the action with a certain prob
                pred_probs = softmax(-pred_costs)
                pred_label = np.random.choice(len(action_list), 1, p=pred_probs)[0]
                pred_action = action_list[pred_label]

                # get the next image, check collision and goal
                next_im = test_data.scene_annotations[scene[0]][current_im][pred_action]
                if next_im=='':
                    image_seq.append(current_im)
                else:
                    image_seq.append(next_im)
                action_seq.append(pred_action)
                print(t, current_im, pred_action, next_im)
                if not(next_im==''): # not collision case
                    collision = 0
                    # check for goal
                    path_dist = len(nx.shortest_path(test_data.graphs_dict[target_lbl][scene[0]], next_im, "goal")) - 2
                    if path_dist <= parIL.steps_from_goal: # GOAL!
                        acc += 1
                        epi_length += t
                        path_ratio += t/float(shortest_path_length) # ratio of estimated path towards shortest path
                        done=1
                        break
                    # get next state from mapNet
                    batch_next, obsv_batch_next = test_data.get_step_data(next_ims=[next_im], scenes=scene, scales=scale)
                    if parIL.use_p_gt:
                        next_im_abs_pose = dh.get_image_poses(info, im_names_all, [next_im], scale[0])
                        abs_poses = np.concatenate((start_abs_pose, next_im_abs_pose), axis=0)
                        rel_poses = dh.relative_poses(poses=abs_poses)
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
                    collision = 1
                    if parIL.stop_on_collision:
                        break
                    if parIL.use_ego_obsv:
                        state = (state[0], state[1], state[2], torch.tensor([collision], dtype=torch.float32).cuda(), state[4])
                    else:
                        state = (state[0], state[1], state[2], torch.tensor([collision], dtype=torch.float32).cuda())
                
            episode_results[i] = (image_seq, action_seq, target_lbl, done) 
        # store the episodes
        episode_results_path = parIL.model_dir+'episode_results_eval_'+str(test_iter)+'.pkl'
        with open(episode_results_path, 'wb') as f:
            pickle.dump(episode_results, f)
        
        success_rate = acc / float(len(test_ids))
        if acc > 0:
            mean_epi_length = epi_length / float(acc)
            avg_path_length_ratio = path_ratio / float(acc)
        else:
            mean_epi_length = 0
            avg_path_length_ratio = 0
        print("Test iter:", test_iter, "Success rate:", success_rate)
        print("Mean epi length:", mean_epi_length, "Avg path length ratio:", avg_path_length_ratio)



if __name__ == '__main__':
    parMapNet = ParametersMapNet()
    parIL = Parameters_IL()
    action_list = np.asarray(parMapNet.action_list)

    if parIL.use_predefined_test_set:
        # Open predefined initial configurations and load them in avd
        avd = AVD_online(par=parIL, nStartPos=0, scene_list=parIL.predefined_test_scenes, 
                                                action_list=action_list, init_configs=parIL.predefined_confs_file)
    else:
        # sample random starting positions and targets from the AVD_online class
        avd = AVD_online(par=parIL, nStartPos=10, scene_list=["Home_001_1"], action_list=action_list)

    test_ids = list(range(len(avd)))

    # Need to load the trained MapNet
    if parIL.finetune_mapNet: # choose whether to use a finetuned mapNet model or not
        mapNet_model = hl.load_model(model_dir=parIL.model_dir, model_name="MapNet", test_iter=parIL.test_iters)
    else:
        mapNet_model = hl.load_model(model_dir=parIL.mapNet_model_dir, model_name="MapNet", test_iter=parIL.mapNet_iters)
    
    if parIL.use_ego_obsv:
        ego_encoder = Encoder()
        ego_encoder.cuda()
        ego_encoder.eval()
    else:
        ego_encoder = None

    evaluate_NavNet(parIL, parMapNet, mapNet_model, ego_encoder, test_iter=parIL.test_iters, 
                                            test_ids=test_ids, test_data=avd, action_list=action_list) 
                     
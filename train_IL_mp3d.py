import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import math
import random
import pickle
from dataloader_mp3d import MP3D_IL, MP3D_online
from IL_Net import ILNet, Encoder
from mapNet import MapNet
from parameters_mp3d import ParametersIL_MP3D, ParametersMapNet_MP3D
import helper as hl
import data_helper as dh
import data_helper_mp3d as dhm
#from QNet import DQN, DQN_2, Encoder
from itertools import chain
from copy import deepcopy
from test_IL import evaluate_ILNet

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_minibatch(batch_size, tvec_dim, seq_len, nActions, data, ex_ids, data_index):
    imgs_batch = torch.zeros(batch_size, seq_len, 3, data.cropSize[1], data.cropSize[0]).float().cuda()
    sseg_batch = torch.zeros(batch_size, seq_len, 1, data.cropSize[1], data.cropSize[0]).float().cuda()
    dets_batch = torch.zeros(batch_size, seq_len, data.dets_nClasses, data.cropSize[1], data.cropSize[0]).float().cuda()
    imgs_obsv_batch = torch.zeros(batch_size, seq_len, 3, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    dets_obsv_batch = torch.zeros(batch_size, seq_len, 1, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    tvec_batch = torch.zeros(batch_size, tvec_dim).float().cuda()
    pose_gt_batch = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    collisions_batch = torch.zeros(batch_size, seq_len).float().cuda()
    costs_batch = torch.zeros(batch_size, seq_len, nActions).float().cuda()
    points2D_batch, local3D_batch = [], []
    image_names, scenes, scales, actions = [], [], [], []
    for k in range(batch_size):
        ex = data[ex_ids[data_index+k]]
        imgs_batch[k,:,:,:,:] = ex["images"]
        sseg_batch[k,:,:,:,:] = ex["sseg"]
        dets_batch[k,:,:,:,:] = ex['dets']
        imgs_obsv_batch[k,:,:,:,:] = ex['images_obsv']
        dets_obsv_batch[k,:,:,:,:] = ex['dets_obsv']
        points2D_batch.append(ex["points2D"]) # nested list of batch_size x n_points x 2
        local3D_batch.append(ex["local3D"]) # nested list of batch_size x n_points x 3
        # Label of the target object for each episode
        tvec_batch[k,ex["target_lbl"]] = 1
        # We need to keep other info to allow us to do the steps later
        image_names.append(ex['images_names'])
        scenes.append(ex['scene'])
        scales.append(ex['scale'])
        pose_gt_batch[k,:,:] = ex["pose"]
        collisions_batch[k,:] = ex['collisions']
        actions.append(ex['actions'])
        costs_batch[k,:,:] = ex['costs']
    mapNet_batch = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, dets_batch, pose_gt_batch)
    IL_batch = (imgs_obsv_batch, dets_obsv_batch, tvec_batch, collisions_batch, actions, costs_batch, image_names, scenes, scales)
    return mapNet_batch, IL_batch


def unroll_policy(parIL, parMapNet, policy_net, mapNet, action_list, batch_size, seq_len, graphs):
    with torch.no_grad():
        nScenes = 4 # how many scenes to use for this minibatch
        ind = np.random.randint(len(parIL.train_scene_list), size=nScenes)
        scene_list = np.asarray(parIL.train_scene_list)
        sel_scene = scene_list[ind]
        mp3d_dagger = MP3D_online(par=parIL, nStartPos=batch_size/nScenes, 
                                        scene_list=sel_scene, action_list=action_list, graphs_dict=graphs)
        #print(len(mp3d_dagger))
        ########### initialize all the arrays to be returned 
        imgs_batch = torch.zeros(batch_size, seq_len, 3, mp3d_dagger.cropSize[1], mp3d_dagger.cropSize[0]).float().cuda()
        sseg_batch = torch.zeros(batch_size, seq_len, 1, mp3d_dagger.cropSize[1], mp3d_dagger.cropSize[0]).float().cuda()
        dets_batch = torch.zeros(batch_size, seq_len, mp3d_dagger.dets_nClasses, mp3d_dagger.cropSize[1], mp3d_dagger.cropSize[0]).float().cuda()
        imgs_obsv_batch = torch.zeros(batch_size, seq_len, 3, mp3d_dagger.cropSizeObsv[1], mp3d_dagger.cropSizeObsv[0]).float().cuda()
        dets_obsv_batch = torch.zeros(batch_size, seq_len, 1, mp3d_dagger.cropSizeObsv[1], mp3d_dagger.cropSizeObsv[0]).float().cuda()
        tvec_batch = torch.zeros(batch_size, parIL.nTargets).float().cuda()
        pose_gt_batch = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
        collisions_batch = torch.zeros(batch_size, seq_len).float().cuda()
        costs_batch = torch.zeros(batch_size, seq_len, len(action_list)).float().cuda()
        points2D_batch, local3D_batch = [], []
        image_names_batch, scene_batch, scale_batch, actions = [], [], [], []
        #########################################
        for i in range(len(mp3d_dagger)): # batch_size
            ex = mp3d_dagger[i]
            img = ex["image"].unsqueeze(0)
            points2D_seq, local3D_seq = [], [] 
            points2D_seq.append(ex["points2D"])
            local3D_seq.append(ex["local3D"])
            sseg = ex["sseg"].unsqueeze(0)
            dets = ex['dets'].unsqueeze(0)
            mapNet_input_start = (img.cuda(), points2D_seq, local3D_seq, sseg.cuda(), dets.cuda())
            # get all other info needed for the episode
            target_lbl = ex["target_lbl"]
            im_obsv = ex['image_obsv'].cuda()
            dets_obsv = ex['dets_obsv'].cuda()
            tvec = torch.zeros(1, parIL.nTargets).float().cuda()
            tvec[0,target_lbl] = 1
            image_name_seq = []
            image_name_seq.append(ex['image_name'])
            scene = ex['scene']
            scene_batch.append(scene)
            scale = ex['scale']
            scale_batch.append(scale)
            graph = mp3d_dagger.graphs_dict[target_lbl][scene]
            abs_pose_seq = np.zeros((seq_len, 3), dtype=np.float32)

            scene_folder = '{}/{}'.format(parIL.mp3d_root, scene)
            info = dhm.load_scene_info(scene_folder)
            im_names_all = list(info.keys()) # info 0 # list of image names in the scene
            im_names_all = np.hstack(im_names_all) # flatten the array
            start_abs_pose = dhm.get_image_poses(info, im_names_all, image_name_seq, scale) # init pose of the episode # 1 x 3 
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
            current_im = image_name_seq[0] #.copy()

            imgs_batch[i,0,:,:,:] = img
            sseg_batch[i,0,:,:,:] = sseg
            dets_batch[i,0,:,:,:] = dets
            imgs_obsv_batch[i,0,:,:,:] = im_obsv
            dets_obsv_batch[i,0,:,:,:] = dets_obsv
            tvec_batch[i] = tvec
            collisions_batch[i,0] = collision_
            abs_pose_seq[0,:] = start_abs_pose
            cost = np.asarray(dhm.get_state_action_cost(current_im, action_list, info, graph), dtype=np.float32)
            costs_batch[i,0,:] = torch.from_numpy(cost).float()

            policy_net.hidden = policy_net.init_hidden(batch_size=1, state_items=len(state)-1)
            for t in range(1, seq_len):
                pred_costs = policy_net(state, parIL.use_ego_obsv) # apply policy for single step
                pred_costs = pred_costs.view(-1).cpu().numpy()
                # use the deterministic version here
                pred_label = np.argmin(pred_costs)
                pred_action = action_list[pred_label]
                actions.append(pred_action)

                # get the next image, check collision and goal
                next_im = mp3d_dagger.scene_infos[scene][current_im][pred_action]
                #image_seq.append(next_im)
                #action_seq.append(pred_action)
                #print(t, current_im, pred_action, next_im)
                if not(next_im==''): # not collision case
                    collision = 0
                    # get next state from mapNet
                    batch_next, obsv_batch_next = mp3d_dagger.get_step_data(next_ims=[next_im], scenes=[scene], scales=[scale])
                    next_im_abs_pose = dhm.get_image_poses(info, im_names_all, [next_im], scale)
                    if parIL.use_p_gt:
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
                    
                    # store the data in the batch
                    (imgs_next, points2D_next, local3D_next, sseg_next, dets_next) = batch_next
                    (imgs_obsv_next, dets_obsv_next) = obsv_batch_next
                    imgs_batch[i,t,:,:,:] = imgs_next
                    sseg_batch[i,t,:,:,:] = sseg_next
                    dets_batch[i,t,:,:,:] = dets_next
                    imgs_obsv_batch[i,t,:,:,:] = imgs_obsv_next
                    dets_obsv_batch[i,t,:,:,:] = dets_obsv_next
                    collisions_batch[i,t] = torch.tensor([collision], dtype=torch.float32)
                    abs_pose_seq[t,:] = next_im_abs_pose
                    cost = np.asarray(dhm.get_state_action_cost(current_im, action_list, info, graph), dtype=np.float32)
                    costs_batch[i,t,:] = torch.from_numpy(cost).float()
                    image_name_seq.append(current_im)
                    points2D_seq.append(points2D_next[0])
                    local3D_seq.append(local3D_next[0])

                else: # collision case
                    #print("Collision!")
                    collision = 1
                    if parIL.stop_on_collision:
                        break
                    if parIL.use_ego_obsv:
                        state = (state[0], state[1], state[2], torch.tensor([collision], dtype=torch.float32).cuda(), state[4])
                    else:
                        state = (state[0], state[1], state[2], torch.tensor([collision], dtype=torch.float32).cuda())

                    # store the data for the collision case (use the ones from the previous step)
                    imgs_batch[i,t,:,:,:] = imgs_batch[i,t-1,:,:,:]
                    sseg_batch[i,t,:,:,:] = sseg_batch[i,t-1,:,:,:]
                    dets_batch[i,t,:,:,:] = dets_batch[i,t-1,:,:,:]
                    imgs_obsv_batch[i,t,:,:,:] = imgs_obsv_batch[i,t-1,:,:,:]
                    dets_obsv_batch[i,t,:,:,:] = dets_obsv_batch[i,t-1,:,:,:]
                    collisions_batch[i,t] = torch.tensor([collision], dtype=torch.float32)
                    abs_pose_seq[t,:] = abs_pose_seq[t-1,:]
                    costs_batch[i,t,:] = costs_batch[i,t-1,:]
                    image_name_seq.append(current_im)
                    points2D_seq.append(points2D_seq[t-1])
                    local3D_seq.append(local3D_seq[t-1])

            # Do the relative pose
            pose_gt_batch[i] = dhm.relative_poses(poses=abs_pose_seq)

            # add the remaining batch data where necessary (i.e. lists)
            image_names_batch.append(image_name_seq)
            points2D_batch.append(points2D_seq)
            local3D_batch.append(local3D_seq)

        # return here
        actions = np.asarray(actions)
        image_names_batch = np.asarray(image_names_batch)

        mapNet_batch = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, dets_batch, pose_gt_batch)
        IL_batch = (imgs_obsv_batch, dets_obsv_batch, tvec_batch, collisions_batch, actions, costs_batch, image_names_batch, scene_batch, scale_batch)
        return mapNet_batch, IL_batch
            

# Choose how to sample the next minibatch
def select_minibatch(par, iters_done):
    sample = random.random()
    eps_threshold = par.EPS_END + (par.EPS_START-par.EPS_END) * math.exp(-1. * iters_done / par.EPS_DECAY)    
    if sample > eps_threshold:
        return 0
    else:
        return 1


def run_mapNet(parMapNet, mapNet, start_info, use_p_gt, pose_gt_batch):
    if use_p_gt:
        p_gt_batch = dh.build_p_gt(parMapNet, pose_gt_batch)
        p_, map_ = mapNet(local_info=start_info, update_type=parMapNet.update_type, 
                                    input_flags=parMapNet.input_flags, p_gt=p_gt_batch)
        p_ = p_gt_batch.clone() # overwrite the predicted with the ground-truth location
    else:
        p_, map_ = mapNet(local_info=start_info, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags)
    return p_, map_



if __name__ == '__main__':
    parMapNet = ParametersMapNet_MP3D()
    parIL = ParametersIL_MP3D()

    action_list = np.asarray(parMapNet.action_list)

    # init the model
    policy_net = ILNet(parIL, parMapNet.map_embedding, parMapNet.orientations, parIL.nTargets, len(action_list), parIL.use_ego_obsv)
    policy_net.train()
    policy_net.cuda()

    # Need to load the trained MapNet
    state_model = hl.load_model(model_dir=parIL.mapNet_model_dir, model_name="MapNet", 
                                    test_iter=parIL.mapNet_iters, eval=not(parIL.finetune_mapNet))
    # If we are not using a trained mapNet model then define a new one
    #state_model = MapNet(parMapNet, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags) #Encoder(par)
    #state_model.cuda()
    #state_model.eval()

    if parIL.finetune_mapNet: # need to chain the parameters of mapNet and policy
        all_params = chain(policy_net.parameters(), state_model.parameters())
    else:
        all_params = policy_net.parameters()
    optimizer = optim.Adam(all_params, lr=parIL.lr_rate)
    scheduler = StepLR(optimizer, step_size=parIL.step_size, gamma=parIL.gamma)


    if parIL.use_ego_obsv:
        ego_encoder = Encoder()
        ego_encoder.cuda()
        ego_encoder.eval()

    # Collect the training episodes
    print("Loading training episodes...")
    mp3d = MP3D_IL(par=parIL, seq_len=parIL.seq_len, nEpisodes=parIL.epi_per_scene, 
                                scene_list=parIL.train_scene_list, action_list=action_list)

    # Need to separately collect the test episodes using mp3d online
    print("Loading test episodes...")
    mp3d_test = MP3D_online(par=parIL, nStartPos=10, 
                                    scene_list=parIL.train_scene_list, action_list=action_list)
    test_ids = list(range(len(mp3d_test)))

    #all_ids = list(range(len(mp3d)))
    #test_ids = all_ids[::100] # select a small subset for testing
    #train_ids = list(set(all_ids) - set(test_ids)) # the rest for training
    train_ids = list(range(len(mp3d)))
    hl.save_params(parIL, parIL.model_dir, name="IL")
    hl.save_params(parMapNet, parIL.model_dir, name="mapNet")
    log = open(parIL.model_dir+"train_log_"+parIL.model_id+".txt", 'w')
    nData = len(train_ids)
    iters_per_epoch = int(nData / float(parIL.batch_size))
    log.write("Iters_per_epoch:"+str(iters_per_epoch)+"\n")
    print("Iters per epoch:", iters_per_epoch)
    loss_list = []
    for ep in range(parIL.nEpochs):
        scheduler.step()
        random.shuffle(train_ids)
        data_index = 0

        for i in range(iters_per_epoch):
            iters = i + ep*iters_per_epoch # actual number of iterations given how many epochs passed

            ch = select_minibatch(par=parIL, iters_done=iters)
            if ch:
                # Sample from the pre-selected episodes, which include random and shortest path sequences
                mapNet_batch, IL_batch = get_minibatch(batch_size=parIL.batch_size, tvec_dim=parIL.nTargets, 
                                                            seq_len=parIL.seq_len, nActions=len(action_list),
                                                            data=mp3d, ex_ids=train_ids, data_index=data_index)
            else:
                # Sample episodes by unrolling the policy to generate the sequence
                mapNet_batch, IL_batch = unroll_policy(parIL, parMapNet, policy_net, state_model,
                                                    action_list, batch_size=parIL.batch_size, 
                                                    seq_len=parIL.seq_len, graphs=mp3d.graphs_dict)

            (imgs_batch, points2D_batch, local3D_batch, sseg_batch, dets_batch, pose_gt_batch) = mapNet_batch
            (imgs_obsv_batch, dets_obsv_batch, tvec_batch, collisions_batch, actions, costs_batch, image_names, scenes, scales) = IL_batch
            data_index += parIL.batch_size

            # get the map for every step from mapNet
            start_info = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, dets_batch)
            if parIL.finetune_mapNet:
                p_, map_ = run_mapNet(parMapNet, state_model, start_info, parIL.use_p_gt, pose_gt_batch)
            else:
                with torch.no_grad():
                    p_, map_ = run_mapNet(parMapNet, state_model, start_info, parIL.use_p_gt, pose_gt_batch)

            '''
            start_info = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, dets_batch)
            if parIL.use_p_gt:
                p_gt_batch = dh.build_p_gt(parMapNet, pose_gt_batch)
                p_, map_ = state_model(local_info=start_info, update_type=parMapNet.update_type, 
                                            input_flags=parMapNet.input_flags, p_gt=p_gt_batch)
                p_ = p_gt_batch.clone() # overwrite the predicted with the ground-truth location
            else:
                p_, map_ = state_model(local_info=start_info, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags)
            '''

            if parIL.use_ego_obsv: # Get the encoding of the img/det in case you add it into the state
                with torch.no_grad():
                    enc_in = torch.cat((imgs_obsv_batch, dets_obsv_batch), 2)
                    enc_in = enc_in.view(parIL.batch_size*parIL.seq_len, 4, parIL.crop_size_obsv[1], parIL.crop_size_obsv[0])
                    ego_obsv_feat = ego_encoder(enc_in) # (b*seq) x 512 x 1 x 1
                    ego_obsv_feat = ego_obsv_feat.view(parIL.batch_size, parIL.seq_len, ego_obsv_feat.shape[1])
                state = (map_, p_, tvec_batch, collisions_batch, ego_obsv_feat)
            else: # state that goes in the IL net is: (map, p, tvec, collision)
                state = (map_, p_, tvec_batch, collisions_batch)

            policy_net.hidden = policy_net.init_hidden(parIL.batch_size, state_items=len(state)-1)
            pred_costs = policy_net(state, parIL.use_ego_obsv)

            loss = policy_net.build_loss(cost_pred=pred_costs, cost_gt=costs_batch, loss_weight=parIL.loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Show, plot, save, test
            if iters % parIL.show_interval == 0:
                log.write("Epoch:"+str(ep)+" ITER:"+str(iters)+" Loss:"+str(loss.data.item())+"\n")
                print("Epoch:", str(ep), " ITER:", str(iters), " Loss:", str(loss.data.item()))
                log.flush()

            if iters > 0:
                loss_list.append(loss.data.item())
            if iters % parIL.plot_interval == 0 and iters>0:
                hl.plot_loss(loss=loss_list, epoch=ep, iteration=iters, step=1, loss_name="L1", loss_dir=parIL.model_dir)

            if iters % parIL.save_interval == 0:
                hl.save_model(model=policy_net, model_dir=parIL.model_dir, model_name="ILNet", train_iter=iters)
                if parIL.finetune_mapNet:
                    hl.save_model(model=state_model, model_dir=parIL.model_dir, model_name="MapNet", train_iter=iters)

            if iters % parIL.test_interval == 0:
                evaluate_ILNet(parIL, parMapNet, mapNet=state_model, ego_encoder=ego_encoder, test_iter=iters, 
                                                test_ids=test_ids, test_data=mp3d_test, action_list=action_list)

            
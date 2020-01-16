import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import data_helper as dh
import helper as hl
from parameters import ParametersMapNet
from mapNet import MapNet
from dataloader import AVD
import time
import pickle


def get_pose(par, p):
    # get the location and orientation of the max value
    # p is r x h x w
    p_tmp = p.view(-1)
    m, _ = torch.max(p_tmp, 0)
    p_tmp = p_tmp.view(par.orientations, par.global_map_dim[0], par.global_map_dim[1])
    p_tmp = p_tmp.detach().cpu().numpy()
    inds = np.where(p_tmp==m.data.item())
    r = inds[0][0] # discretized orientation
    zb = inds[1][0]
    xb = inds[2][0]
    return r, zb, xb


def undo_discretization(par, zb, xb):
    # Transform grid coords back to 3D
    x = (xb-(par.global_map_dim[0]-1)/2.0) * par.cell_size
    z = (zb-(par.global_map_dim[1]-1)/2.0) * par.cell_size
    return z, x


def evaluate_MapNet(par, test_iter, test_ids, test_data):
    print("\nRunning validation on MapNet!")
    with torch.no_grad():
        # Load the model
        test_model = hl.load_model(model_dir=par.model_dir, model_name="MapNet", test_iter=test_iter)
        episode_results = {} # store predictions and ground-truth in order to visualize
        error_list=[]
        angle_acc = 0
        for i in test_ids:
            test_ex = test_data[i]
            imgs_seq = test_ex["images"]
            imgs_name = test_ex["images_names"]
            points2D_seq = test_ex["points2D"]
            local3D_seq = test_ex["local3D"]
            pose_gt_seq = test_ex["pose"]
            abs_pose_gt_seq = test_ex["abs_pose"]
            sseg_seq = test_ex["sseg"]
            dets_seq = test_ex["dets"]
            scene = test_ex["scene"]
            scale = test_ex["scale"]
            imgs_batch = imgs_seq.unsqueeze(0)
            pose_gt_batch = np.expand_dims(pose_gt_seq, axis=0)
            sseg_batch = sseg_seq.unsqueeze(0)
            dets_batch = dets_seq.unsqueeze(0)
            points2D_batch, local3D_batch = [], [] # add another dimension for the batch
            points2D_batch.append(points2D_seq)
            local3D_batch.append(local3D_seq)

            local_info = (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), dets_batch.cuda())
            p_pred, map_pred = test_model(local_info, update_type=par.update_type, input_flags=par.input_flags)
            # remove the tensors from gpu memory
            p_pred = p_pred.cpu().detach()
            map_pred = map_pred.cpu().detach()
            # Remove the first step in any sequence since it is a constant
            p_pred = p_pred[:,1:,:,:,:]
            pose_gt_batch = pose_gt_batch[:,1:,:]
            pred_pose = np.zeros((par.seq_len, 3), dtype=np.float32)
            episode_error=[] # put the errors of the episode
            for s in range(p_pred.shape[1]): # seq_len-1
                # convert p to coordinates and orientation values
                _, zb, xb = get_pose(par, p=p_pred[0,s,:,:,:])
                z_pred, x_pred = undo_discretization(par, zb, xb)
                # get the error
                pred_coords = np.array([x_pred, z_pred], dtype=np.float32)
                gt_coords = pose_gt_batch[0,s,:2]
                error = np.linalg.norm( gt_coords - pred_coords )
                episode_error.append(error)

                # store predictions and gt
                pred_pose[s+1, :] = np.array([x_pred, z_pred, pose_gt_batch[0,s,2]], dtype=np.float32)
            
            episode_results[i] = (imgs_name, pose_gt_seq, abs_pose_gt_seq, pred_pose, scene, scale)

            episode_error = np.asarray(episode_error) 
            error_list.append( np.median(episode_error) )

        with open(par.model_dir+'episode_results_'+str(test_iter)+'.pkl', 'wb') as f:    
            pickle.dump(episode_results, f)
    
        error_list = np.asarray(error_list)
        error_res = error_list.mean()
        print("Test_iter:", test_iter, "Position_error:", error_res, "Seq_len:", par.seq_len)


if __name__ == '__main__':
    par = ParametersMapNet()
    print("Loading the test data...")
    avd_test = AVD(par, seq_len=par.seq_len, nEpisodes=50, scene_list=par.test_scene_list, action_list=par.action_list)    
    test_ids = list(range(len(avd_test)))
    evaluate_MapNet(par, test_iter=par.test_iter, test_ids=test_ids, test_data=avd_test)

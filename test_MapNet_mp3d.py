
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import random
import cv2
import math
import matplotlib.pyplot as plt
#import data_helper as dh
import helper as hl
#from parameters import Parameters
from parameters_mp3d import ParametersMapNet_MP3D
from mapNet import MapNet
from dataloader_mp3d import MP3D
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
    #zb = (par.global_map_dim[1]-1)-zb
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
        # save episodes 
        # ** Need to switch to a batch size **
        for i in test_ids:
            #print(i)
            #start = time.time()
            test_ex = test_data[i]
            imgs_seq = test_ex["images"]
            imgs_name = test_ex["images_names"]
            points2D_seq = test_ex["points2D"]
            local3D_seq = test_ex["local3D"]
            pose_gt_seq = test_ex["pose"]
            abs_pose_gt_seq = test_ex["abs_pose"]
            #print(pose_gt_seq)
            sseg_seq = test_ex["sseg"]
            dets_seq = test_ex["dets"]
            scene = test_ex["scene"]
            scale = test_ex["scale"]
            # for now assume that test_batch_size=1
            imgs_batch = imgs_seq.unsqueeze(0)
            pose_gt_batch = np.expand_dims(pose_gt_seq, axis=0)
            sseg_batch = sseg_seq.unsqueeze(0)
            dets_batch = dets_seq.unsqueeze(0)
            points2D_batch, local3D_batch = [], [] # add another dimension for the batch
            points2D_batch.append(points2D_seq)
            local3D_batch.append(local3D_seq)
            #p_gt_batch = dh.build_p_gt(par, pose_gt_batch)
            #print("Input time:", time.time()-start)
            #print('scene: {}'.format(scene))
            #print('imgs_name : {}'.format(imgs_name))
            #start = time.time()
            local_info = (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), dets_batch.cuda())
            p_pred, map_pred = test_model(local_info, update_type=par.update_type, input_flags=par.input_flags)
            # remove the tensors from gpu memory
            p_pred = p_pred.cpu().detach()
            map_pred = map_pred.cpu().detach()
            #p_gt_batch = p_gt_batch.cpu().detach()
            #print("MapNet time:", time.time()-start)
            #for j in range(par.orientations):
            #    angle = 2*np.pi*(j/par.orientations)
            #    print(j, angle)
            #start = time.time()
            # Remove the first step in any sequence since it is a constant
            p_pred = p_pred[:,1:,:,:,:]
            #p_gt_batch = p_gt_batch[:,1:,:,:,:]
            pose_gt_batch = pose_gt_batch[:,1:,:]
            #print(p_pred.shape)
            pred_pose = np.zeros((par.seq_len, 3), dtype=np.float32)
            episode_error=[] # add the errors of the episode so you can do the median
            for s in range(p_pred.shape[1]): # seq_len-1
                # convert p to coordinates and orientation values
                rb, zb, xb = get_pose(par, p=p_pred[0,s,:,:,:])
                # ** Need to figure out how to make r_gt into an actual value to use for visualization and median error
                r_gt = np.floor( np.mod(pose_gt_batch[0,s,2]/(2*np.pi), 1) * par.orientations )
                #print(rb, r_gt)
                #r_gt, zb_gt, xb_gt = get_pose(par, p=p_gt_batch[0,s,:,:,:])
                #print("est:",xb,zb, " gt:", xb_gt,zb_gt)
                # undiscretize the map coords
                z_pred, x_pred = undo_discretization(par, zb, xb)
                #z_gt, x_gt = undo_discretization(par, zb_gt, xb_gt)
                #print("est:", x_pred, z_pred, " gt:", x_gt, z_gt)
                # get the error
                #pred_coords = np.array([z_pred, x_pred], dtype=np.float32)
                pred_coords = np.array([x_pred, z_pred], dtype=np.float32)
                #gt_coords = np.array([z_gt, x_gt], dtype=np.float32)
                #print(pose_gt_batch.shape)
                gt_coords = pose_gt_batch[0,s,:2]
                #print(pred_coords, gt_coords)
                error = np.linalg.norm( gt_coords - pred_coords )
                episode_error.append(error)
                # estimate the angle accuracy
                if rb==r_gt:
                    angle_acc+=1

                # store predictions and gt
                pred_pose[s+1, :] = np.array([x_pred, z_pred, pose_gt_batch[0,s,2]], dtype=np.float32) # ** for now use gt orienation
            #print(pred_pose)
            episode_results[i] = (imgs_name, pose_gt_seq, abs_pose_gt_seq, pred_pose, scene, scale)

            episode_error = np.asarray(episode_error) 
            error_list.append( np.median(episode_error) )
            #print("Metrics time:", time.time()-start)

        #with open('examples/MapNet/episode_results.pkl', 'wb') as f:
        with open(par.model_dir+'episode_results_'+str(test_iter)+'.pkl', 'wb') as f:    
            pickle.dump(episode_results, f)
    
        error_list = np.asarray(error_list)
        #print(error_list)
        error_res = error_list.mean()
        angle_acc = angle_acc / float(len(test_ids)) # ** need to change this to number of steps
        print("Test iter:", test_iter, "Position error:", error_res, "Angle accuracy:", angle_acc)
        res_file = open(par.model_dir+"val_"+par.model_id+".txt", "a+")
        res_file.write("Test iter:" + str(test_iter) + "\n")
        res_file.write("Test set:" + str(len(test_ids)) + "\n")
        res_file.write("Position error:" + str(error_res) + "\n")
        res_file.write("Angle accuracy:" + str(angle_acc) + "\n")
        res_file.write("\n")
        res_file.close()

#'''
if __name__ == '__main__':
    par = ParametersMapNet_MP3D()
    '''
    # create dataset
    mp3d_test = MP3D(par, seq_len=par.seq_len, nEpisodes=50, scene_list=par.test_scene_list, action_list=par.action_list)
    # save the test data for reproducibility
    mp3d_file = open('{}/test_mapNet_{}.pkl'.format(par.mp3d_root, par.seq_len), 'wb')
    pickle.dump(mp3d_test, mp3d_file)
    '''


    print("Loading the test data...")
    mp3d_test = pickle.load(open('{}/test_mapNet_{}.pkl'.format(par.mp3d_root, par.seq_len), 'rb'))
    test_ids = list(range(len(mp3d_test)))
    evaluate_MapNet(par, test_iter=2500, test_ids=test_ids, test_data=mp3d_test)
    evaluate_MapNet(par, test_iter=3500, test_ids=test_ids, test_data=mp3d_test)
#'''










'''
# The returned image feat are 1 x 512 x 23 x 40, much smaller than the image. This depends on how shallow the network is.
# Three choices: 
# 1) Do the ground projection given the current size
# 2) Upscale the features to match the img size
# 3) Change all resNet stride parameters to 1

# Resize the features to the image/depth resolution
img_feat_resized = F.interpolate(img_feat, size=(par.crop_size[1], par.crop_size[0]), mode='nearest')
print(img_feat_resized.shape)

# Create the grid and discretize the set of coordinates into the bins
# Points2D holds the image pixel coordinates with valid depth values
# Local3D holds the X,Y,Z coordinates that correspond to the points2D

# For each local3d find which bin it belongs to
valid = []
map_coords = np.zeros((local3D.shape[0], 2), dtype=np.int)
#map_occ = np.zeros((par.map_dim[0], par.map_dim[1]), dtype=np.float32)
for i in range(local3D.shape[0]):
    x, z = local3D[i,0], local3D[i,2]
    xb = int( math.floor(x/par.cell_size) + (par.map_dim[0]-1)/2.0 )
    zb = int( math.floor(z/par.cell_size) + (par.map_dim[1]-1)/2.0 )
    map_coords[i,0] = xb
    zb = (par.map_dim[1]-1)-zb # mirror the z axis so that the origin is at the bottom
    map_coords[i,1] = zb
    # keep bin coords within dimensions
    #if xb<0 or zb<0 or xb>=par.map_dim[0] or zb>=par.map_dim[1]:
    if xb>=0 and zb>=0 and xb<par.map_dim[0] and zb<par.map_dim[1]:
        valid.append(i)
        #map_occ[zb,xb] = 1
        #invalid.append(i)
        #print(x, z, xb, zb)

#plt.imshow(map_occ)
#plt.show()
#raise Exception("gggg")

valid = np.asarray(valid, dtype=np.int)
#print(invalid)
#print(local3D.shape)
points2D = points2D[valid,:]
local3D = local3D[valid,:]
map_coords = map_coords[valid,:]
#print(local3D.shape)


feat_dict = {} # keep track of the feature vectors that project to each map location

# go through each point in points2D and collect the feature from the img_feat_resized
# add it to the right map_coord in the map. if the location is not empty, then do max pooling 
# of the current feature with the existing one (channel wise) 
for i in range(points2D.shape[0]):
    pix_x, pix_y = points2D[i,0], points2D[i,1]
    map_x, map_y = map_coords[i,0], map_coords[i,1]
    #print(pix_x, pix_y, map_x, map_y)
    pix_feat = img_feat_resized[0, :, pix_y, pix_x]
    #print(pix_feat.shape)
    if (map_x,map_y) not in feat_dict.keys():
        feat_dict[(map_x,map_y)] = pix_feat.unsqueeze(0)
    else:
        feat_dict[(map_x,map_y)] = torch.cat(( feat_dict[(map_x,map_y)], pix_feat.unsqueeze(0) ), 0) # cat with the existing features in that bin
        #print(feat_dict[(map_x,map_y)].shape)

    # check whether the location already contains features
    #grid_feat = grid[map_x, map_y, :] # in the beginning all zeroes
    #print(torch.sum(grid_feat))
    #if torch.sum(grid_feat)==0: # if all zeroes, then just add the features as they are
    #    grid[map_x, map_y, :] = pix_feat
    #else:
    #print(torch.sum(grid_feat))
#print(feat_dict[(map_x,map_y)].shape)

grid = np.zeros((par.map_dim[0], par.map_dim[1], img_feat_resized.shape[1]), dtype=np.float32)
grid = torch.tensor(grid, dtype=torch.float32)
# Do max-pooling over the bins. Is the max-pooling being done channel-wise?
for (map_x,map_y) in feat_dict.keys():
    bin_feats = feat_dict[(map_x,map_y)] # [n x d] n:number of feature vectors projected, d:feat_dim
    bin_feat, ind = torch.max(bin_feats,0) # [1 x d]
    grid[map_y, map_x, :] = bin_feat

    raise Exception("gggg")

# Pass the grid to a CNN to get the observation with 32-D embeddings
'''

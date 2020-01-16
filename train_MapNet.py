
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import random
import data_helper as dh
import helper as hl
from mapNet import MapNet
from parameters import ParametersMapNet
from dataloader import AVD
from test_MapNet import evaluate_MapNet


def get_minibatch(par, data, ex_ids, data_index):
    imgs_batch = torch.zeros(par.batch_size, par.seq_len, 3, par.crop_size[1], par.crop_size[0])
    pose_gt_batch = np.zeros((par.batch_size, par.seq_len, 3), dtype=np.float32)
    sseg_batch = torch.zeros(par.batch_size, par.seq_len, 1, par.crop_size[1], par.crop_size[0])
    dets_batch = torch.zeros(par.batch_size, par.seq_len, par.dets_nClasses, par.crop_size[1], par.crop_size[0])
    points2D_batch, local3D_batch = [], []
    for k in range(par.batch_size):
        ex = data[ex_ids[data_index+k]] # episode
        imgs_seq = ex["images"]
        points2D_seq = ex["points2D"]
        local3D_seq = ex["local3D"]
        pose_gt_seq = ex["pose"]
        sseg_seq = ex["sseg"]
        dets_seq = ex["dets"]
        imgs_batch[k,:,:,:,:] = imgs_seq
        pose_gt_batch[k,:,:] = pose_gt_seq
        sseg_batch[k,:,:,:,:] = sseg_seq
        dets_batch[k,:,:,:,:] = dets_seq
        points2D_batch.append(points2D_seq) # nested list of batch_size x seq_len x n_points x 2
        local3D_batch.append(local3D_seq) # nested list of batch_size x seq_len x n_points x 3
    return (imgs_batch.cuda(), points2D_batch, local3D_batch, pose_gt_batch, sseg_batch.cuda(), dets_batch.cuda())


if __name__ == '__main__':
    par = ParametersMapNet()
    # Init the model
    mapNet_model = MapNet(par, update_type=par.update_type, input_flags=par.input_flags) #Encoder(par)
    mapNet_model.cuda()
    mapNet_model.train()
    optimizer = optim.Adam(mapNet_model.parameters(), lr=par.lr_rate)
    scheduler = StepLR(optimizer, step_size=par.step_size, gamma=par.gamma)
    # Load the dataset
    print("Loading the training data...")
    avd = AVD(par, seq_len=par.seq_len, nEpisodes=par.epi_per_scene, 
                scene_list=par.train_scene_list, action_list=par.action_list, with_shortest_path=par.with_shortest_path)

    log = open(par.model_dir+"train_log_"+par.model_id+".txt", 'w')
    hl.save_params(par, par.model_dir, name="mapNet")
    loss_list=[]

    all_ids = list(range(len(avd)))
    test_ids = all_ids[::100] # select a small subset for validation
    train_ids = list(set(all_ids) - set(test_ids)) # the rest for training
    
    nData = len(train_ids)
    iters_per_epoch = int(nData / float(par.batch_size))
    log.write("Iters_per_epoch:"+str(iters_per_epoch)+"\n")
    print("Iters per epoch:", iters_per_epoch)
    for ep in range(par.nEpochs):
        scheduler.step()
        random.shuffle(train_ids)
        data_index = 0
        for i in range(iters_per_epoch):
            iters = i + ep*iters_per_epoch # actual number of iterations given how many epochs passed

            # Sample the training minibatch
            batch = get_minibatch(par, data=avd, ex_ids=train_ids, data_index=data_index)
            (imgs_batch, points2D_batch, local3D_batch, pose_gt_batch, sseg_batch, dets_batch) = batch                                                                    
            
            p_gt_batch = dh.build_p_gt(par, pose_gt_batch)
            data_index += par.batch_size

            # Do a forward pass of mapNet
            local_info = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, dets_batch)
            p_pred, map_pred = mapNet_model(local_info, update_type=par.update_type, 
                                                        input_flags=par.input_flags, p_gt=None)

            # Backprop the loss
            loss = mapNet_model.build_loss(p_pred, p_gt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Show, plot, save, test
            if iters % par.show_interval == 0:
                log.write("Epoch:"+str(ep)+" ITER:"+str(iters)+" Loss:"+str(loss.data.item())+"\n")
                print("Epoch:", str(ep), " ITER:", str(iters), " Loss:", str(loss.data.item()))
                log.flush()

            if iters > 0:
                loss_list.append(loss.data.item())
            if iters % par.plot_interval == 0 and iters>0:
                hl.plot_loss(loss=loss_list, epoch=ep, iteration=iters, step=1, loss_name="NLL", loss_dir=par.model_dir)

            if iters % par.save_interval == 0:
                hl.save_model(model=mapNet_model, model_dir=par.model_dir, model_name="MapNet", train_iter=iters)

            if iters % par.test_interval == 0:
                evaluate_MapNet(par, test_iter=iters, test_ids=test_ids, test_data=avd)


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import math
import networkx as nx

def load_scene_info(scene_folder):
    info = np.load('{}/img_act_dict.npy'.format(scene_folder), allow_pickle=True).item()
    return info

def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))
    if normalize:
        src_image = src_image.astype(np.float) / 255.0 #* 2.0 - 1.0
    return src_image

# unproject pixels to the 3D camera coordinate frame
def depth_to_3D(depth, orig_res, crop_res):
    non_zero_inds = np.where(depth>0) # get all non-zero points
    #print('sum(depth) = {}'.format(np.sum(depth)))
    points2D = np.zeros((len(non_zero_inds[0]), 2), dtype=np.int)
    points2D[:,0] = non_zero_inds[1] # inds[1] is x (width coordinate)
    points2D[:,1] = non_zero_inds[0]

    # scale the intrinsics based on the new resolution
    fx, fy, cx, cy = 128.0, 128.0, 128.0, 128.0
    fx *= crop_res[0] / float(orig_res[0])
    fy *= crop_res[1] / float(orig_res[1])
    cx *= crop_res[0] / float(orig_res[0])
    cy *= crop_res[1] / float(orig_res[1])
    #print(fx, fy, cx, cy)
    # scale the depth based on the given AVD scale parameter
    #depth = depth/1000.0 #float(scale)
    
    # unproject the points
    z = depth[points2D[:,1], points2D[:,0]]
    local3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)
    a = points2D[:,0]-cx
    b = points2D[:,1]-cy
    q1 = a[:,np.newaxis]*z[:,np.newaxis] / fx
    q2 = b[:,np.newaxis]*z[:,np.newaxis] / fy
    local3D[:,0] = q1.reshape(q1.shape[0])
    local3D[:,1] = q2.reshape(q2.shape[0])
    local3D[:,2] = z
    return points2D, local3D

## convert sseg into object detection mask
#def generate_detection_image_from_sseg(sseg_img, image_size, num_classes, is_binary=True):



def getImageData(datapath, scene_det_dict, dets_nClasses, im_name, scene, scale, cropSize, orig_res, pixFormat, normalize, get3d=True):
    scene_folder = '{}/{}'.format(datapath, scene)
    ## processing rgb_img
    rgb_img = cv2.imread('{}/{}/{}.jpg'.format(scene_folder, 'images', im_name), 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (cropSize[0],cropSize[1]))
    imgData = convert_image_by_pixformat_normalize(rgb_img, pixFormat, normalize)
    
    if get3d:
        ## processing semantic_img
        sseg_depth_np = np.load('{}/{}/{}.npy'.format(scene_folder, 'others', im_name), allow_pickle=True).item()
        semantic_img = sseg_depth_np['sseg']

        sseg_img = np.float32(semantic_img % 40) # 256 x 256
        sseg_img = cv2.resize(sseg_img, (cropSize[0], cropSize[1]))
        sseg_img = np.expand_dims(sseg_img, axis=0)
        det_img = get_det_mask(im_name, scene_det_dict[scene], cropSize, dets_nClasses)
        #sseg_img = sseg_img.transpose((2, 0, 1))
        #print('sseg_img.shape = {}'.format(sseg_img.shape))
        #print('sseg_img.dtype = {}'.format(sseg_img.dtype))
        
        ## get 2d and 3d points
        depth_img = sseg_depth_np['depth']
        depth_img = cv2.resize(depth_img, (cropSize[0],cropSize[1]))
        points2D, local3D = depth_to_3D(depth_img, orig_res, cropSize)
        
        return imgData, sseg_img, det_img, points2D, local3D
    else:
        det_img = get_det_mask(im_name, scene_det_dict[scene], cropSize, dets_nClasses)
        return imgData, det_img

def create_scene_graph(info, im_names, action_set, goal_im_names=None):
    graph = nx.DiGraph()
    for i in range(im_names.shape[0]):
        graph.add_node(im_names[i])

    #action_set.remove('stop') # remove the last action "stop"
    for i in range(im_names.shape[0]):
        im_info = info[im_names[i]]
        #print im_info
        for action in action_set:
            next_image = im_info[action] # if there is not an action here (i.e. collision), then the value of next_image is ''
            graph.add_edge(im_names[i], next_image, action=action)
    #print graph.edges['000310000010101.jpg', '000310000120101.jpg']['action']
    if goal_im_names is not None:
        # specify which nodes are goal nodes in the graph by adding the action 'stop' to them
        graph.add_node("goal")
        for i in range(len(goal_im_names)):
            graph.add_edge(goal_im_names[i], "goal", action="stop")
    return graph

def get_scene_target_graphs(datasetPath, cat_dict, targets_data, actions):
    graphs_dict = {}
    cats = cat_dict.keys()
    for c in cats:
        cat_scenes = targets_data[c]
        scene_dict = {}
        for scene in cat_scenes.keys():
            scene_folder = '{}/{}'.format(datasetPath, scene)
            info = load_scene_info(scene_folder)
            im_names_all = list(info.keys()) # list of image names in the scene
            im_names_all = np.hstack(im_names_all) # flatten the array
            goal_ims = cat_scenes[scene]
            gr = create_scene_graph(info, im_names_all, actions, goal_im_names=goal_ims)
            #path = nx.shortest_path(gr, im_names_all[0], "goal")
            #print(path)
            scene_dict[scene] = gr
            #print(scene_dict)
        # use the lbl instead of the name to store in the dict
        lbl = cat_dict[c]
        graphs_dict[lbl] = scene_dict    
        #print(graphs_dict)
    return graphs_dict

def relative_poses(poses):
    #print(poses)
    # poses (seq_len x 3) contains the ground-truth camera positions and orientation in the sequence
    # make them relative to the first pose in the sequence
    rel_poses = np.zeros((poses.shape[0], poses.shape[1]), dtype=np.float32)
    x0 = poses[0,0]
    y0 = poses[0,1]
    a0 = poses[0,2]
    # relative translation
    rel_poses[:,0] = poses[:,0] - x0
    rel_poses[:,1] = poses[:,1] - y0
    #print(rel_poses)
    rel_poses[:,2] = poses[:,2] - a0
    #print(rel_poses)
    return rel_poses

# find the targets that exist in the particular scene
def candidate_targets(scene, cat_dict, targets_data):
    cats = cat_dict.keys()
    candidates=[]
    for c in cats:
        if scene in targets_data[c].keys():
            candidates.append(c)
    return candidates

def get_state_action_cost(current_im, actions, info, graph):
    # return the costs of the actions from a certain image
    current_im_cost = len(nx.shortest_path(graph, current_im, "goal"))-2
    act_cost = []
    for act in actions:
        next_im = info[current_im][act]
        if next_im=='':
            cost = 1 #2 # collision cost, should collision have 0 cost?
        else:
            next_im_cost = len(nx.shortest_path(graph, next_im, "goal"))-2
            if next_im_cost<=0: # given the selected action, the next im is a goal
                cost = -2
            else:
                cost = next_im_cost - current_im_cost
        # put a bound on cost
        if cost > 1:
            cost = 1
        act_cost.append(cost)
    return act_cost

def check_if_goal_is_reachable(current_im, graph):
    try:
        nx.shortest_path(graph, current_im, "goal")
        #print('goal is reachable from the initial image.')
        return True
    except:
        #print('goal is not reachable ...')
        return False

def get_image_poses(info, im_names_all, im_names, scale):
    #print(im_names)
    poses = np.zeros((len(im_names),3), dtype=np.float32)
    for i in range(len(im_names)):
        poses[i,:] = np.asarray(info[im_names[i]]['pose'], dtype=np.float32)
    return poses

labels_to_index = {}
labels_to_index[25]=0
labels_to_index[26]=1
labels_to_index[27]=2
labels_to_index[29]=3
labels_to_index[33]=4
labels_to_index[35]=5
labels_to_index[36]=6
labels_to_index[37]=7
labels_to_index[38]=8
labels_to_index[39]=9
labels_to_index[40]=10
labels_to_index[41]=11
labels_to_index[42]=12
labels_to_index[43]=13
labels_to_index[44]=14
labels_to_index[45]=15
labels_to_index[46]=16
labels_to_index[57]=17
labels_to_index[58]=18
labels_to_index[59]=19
labels_to_index[60]=20
labels_to_index[61]=21
labels_to_index[62]=22
labels_to_index[63]=23
labels_to_index[64]=24
labels_to_index[65]=25
labels_to_index[66]=26
labels_to_index[67]=27
labels_to_index[68]=28
labels_to_index[69]=29
labels_to_index[70]=30
labels_to_index[71]=31
labels_to_index[72]=32
labels_to_index[73]=33
labels_to_index[74]=34
labels_to_index[75]=35
labels_to_index[76]=36
labels_to_index[77]=37
labels_to_index[79]=38
labels_to_index[80]=39

def generate_detection_image(detections, image_size, num_classes, is_binary):
    res = np.zeros((num_classes, image_size[1], image_size[0]), dtype=np.float32)
    boxes = detections['bbox']
    labels = detections['labels']
    scores = detections['scores']
    nDets = detections['num_dets']
    for i in range(nDets):
        lbl = labels[i]
        score = scores[i]
        box=boxes[i] # top left bottom right
        x1 = int(box[0]/256*image_size[0])
        y1 = int(box[1]/256*image_size[1])
        x2 = int(box[2]/256*image_size[0])
        y2 = int(box[3]/256*image_size[1])
        #print(y1,x1,y2,x2, labels_to_cats[lbl], score)
        if num_classes==1:
            res[0, y1:y2, x1:x2] = lbl
        else:
            value = score
            if is_binary:
                value = 1
            if lbl in labels_to_index.keys():
                ind = labels_to_index[lbl]
                res[ind, y1:y2, x1:x2] = value
    return res



def get_det_mask(im_name, scene_dets, cropSize, dets_nClasses):
    im_dets = scene_dets[im_name]
    det_mask = generate_detection_image(detections=im_dets, image_size=cropSize, 
        num_classes=dets_nClasses, is_binary=True)
    return det_mask

def load_detections(datasetPath, scene_list):
    scene_det_dict = {}
    for scene in scene_list:
        det_file = np.load('{}/{}/maskrcnn_detections.npy'.format(datasetPath, scene), allow_pickle=True).item()
        scene_det_dict[scene] = det_file
    return scene_det_dict

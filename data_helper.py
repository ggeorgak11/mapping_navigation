import numpy as np
import scipy.io as sio
import os
import json
import cv2
import random
import PIL.Image
import torch
import networkx as nx
from functools import reduce


def load_scene_info(avd_root, scene):
    # annotations contains the action information
    # image_structs contains camera information for each image
    image_structs_path = os.path.join(avd_root, scene, 'image_structs.mat')
    annotation_path = os.path.join(avd_root, scene, 'annotations.json')
    data = sio.loadmat(image_structs_path)
    scale = data['scale'][0][0]
    image_structs = data['image_structs'][0]
    #rot = image_structs['R']
    im_names_all = image_structs['image_name'] # info 0 # list of image names in the scene
    im_names_all = np.hstack(im_names_all) # flatten the array
    world_poses = image_structs['world_pos'] # info 3
    directions = image_structs['direction'] # info 4
    annotations = json.load(open(annotation_path))
    return annotations, scale, im_names_all, world_poses, directions

def invert_pose(R,T):
    R = np.linalg.inv(R)
    T = np.dot(-R, T)
    return R, T

def relative_poses(poses):
    # Poses (seq_len x 3) contains the ground-truth camera positions and orientation in the sequence
    # Make them relative to the first pose in the sequence
    rel_poses = np.zeros((poses.shape[0], poses.shape[1]), dtype=np.float32)
    x0 = poses[0,0]
    y0 = poses[0,1]
    a0 = poses[0,2]
    # relative translation
    rel_poses[:,0] = poses[:,0] - x0
    rel_poses[:,1] = poses[:,1] - y0
    # relative orientation
    rel_poses[:,2] = poses[:,2] - a0
    return rel_poses


def absolute_poses(rel_pose, origin):
    # Inverse to relative_poses()
    # Given the rel poses and the origin pose (i.e. first pose in the episode), get the absolute poses in the episode
    abs_pose = np.zeros((rel_pose.shape[0], 3), dtype=np.float32)
    a0 = origin[2]
    abs_pose[:,0] = rel_pose[:,0] + origin[0]
    abs_pose[:,1] = rel_pose[:,1] + origin[1]    
    abs_pose[:,2] = rel_pose[:,2] + a0
    return abs_pose


def build_p_gt(par, pose_gt_batch):
    # Create the ground-truth pose tensor to be used in the objective function
    batch_size = pose_gt_batch.shape[0]
    seq_len = pose_gt_batch.shape[1]
    p_gt = np.zeros((batch_size, seq_len, par.orientations, par.global_map_dim[0], par.global_map_dim[1]), dtype=np.float32) 
    for b in range(batch_size):
        pose_seq = pose_gt_batch[b,:,:] # seq_len x 3
        map_coords_gt, valid, _ = discretize_coords(x=pose_seq[:,0], z=pose_seq[:,1], map_dim=par.global_map_dim, cell_size=par.cell_size)
        dir_ind = np.floor( np.mod(pose_seq[:,2]/(2*np.pi), 1) * par.orientations ) # use the gt direction to get the orientation index 
        # the indices not included in valid are those outside the map, p_gt is all zeroes in that case
        map_coords_gt = map_coords_gt[valid,:]
        dir_ind = dir_ind[valid]
        p_gt[b, valid, dir_ind.astype(int), map_coords_gt[:,1], map_coords_gt[:,0]] = 1
    return torch.from_numpy(p_gt).cuda().float()  
    

def discretize_coords(x, z, map_dim, cell_size):
    # Discretize the 3D (just x,z) coordinates given the grid and bin dimensions. Part of ground-projection. 
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    map_coords = np.zeros((len(x), 2), dtype=np.int)
    map_occ = np.zeros((1, map_dim[0], map_dim[1]), dtype=np.float32)
    xb = np.floor(x[:]/cell_size) + (map_dim[0]-1)/2.0
    zb = np.floor(z[:]/cell_size) + (map_dim[1]-1)/2.0
    xb = xb.astype(int)
    zb = zb.astype(int)
    zb = (map_dim[1]-1)-zb # flip the z axis
    map_coords[:,0] = xb
    map_coords[:,1] = zb
    # keep bin coords within dimensions
    inds_1 = np.where(xb>=0)
    inds_2 = np.where(zb>=0)
    inds_3 = np.where(xb<map_dim[0])
    inds_4 = np.where(zb<map_dim[1])
    valid = reduce(np.intersect1d, (inds_1, inds_2, inds_3, inds_4))
    xb = xb[valid]
    zb = zb[valid]
    map_occ[0,zb,xb] = 1
    return map_coords, valid, map_occ


def depth_to_3D(depth, intr, orig_res, crop_res):
    # Unproject pixels to the 3D camera coordinate frame
    non_zero_inds = np.where(depth>0) # get all non-zero points
    points2D = np.zeros((len(non_zero_inds[0]), 2), dtype=np.int)
    points2D[:,0] = non_zero_inds[1] # inds[1] is x (width coordinate)
    points2D[:,1] = non_zero_inds[0]
    # scale the intrinsics based on the new resolution
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    fx *= crop_res[0] / float(orig_res[0])
    fy *= crop_res[1] / float(orig_res[1])
    cx *= crop_res[0] / float(orig_res[0])
    cy *= crop_res[1] / float(orig_res[1])
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


def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))
    if normalize:
        src_image = src_image.astype(np.float) / 255.0 #* 2.0 - 1.0
    return src_image


def load_depth_file(scene_path, frame_id):
    depth_id = frame_id[:-1] + "3"
    if depth_id=="001310003930103": # corrupted file
        depth_id = "001310004050103" # use similar image
    depth_data = PIL.Image.open(scene_path+"high_res_depth/"+depth_id+".png")
    depth = np.array(depth_data, dtype=np.int)
    return depth


def create_scene_graph(annotations, im_names, action_set, goal_im_names=None):
    # Creates a graph for a scene where nodes are the images and edges are the actions.
    graph = nx.DiGraph()
    for i in range(im_names.shape[0]):
        graph.add_node(im_names[i])
    for i in range(im_names.shape[0]):
        im_info = annotations[im_names[i]]
        for action in action_set:
            next_image = im_info[action] # if there is not an action here (i.e. collision), then the value of next_image is ''
            graph.add_edge(im_names[i], next_image, action=action)
    if goal_im_names is not None:
        # specify which nodes are goal nodes in the graph by adding the action 'stop' to them
        graph.add_node("goal")
        for i in range(len(goal_im_names)):
            graph.add_edge(goal_im_names[i], "goal", action="stop")
    return graph


def get_scene_target_graphs(datasetPath, cat_dict, targets_data, actions):
    # Wrapper to create graphs for all scenes
    graphs_dict = {}
    cats = cat_dict.keys()
    for c in cats:
        cat_scenes = targets_data[c.encode()]
        scene_dict = {}
        for scene in cat_scenes.keys():
            annotations, _, im_names_all, _, _ = load_scene_info(datasetPath, scene.decode("utf-8"))
            goal_ims = [x.decode("utf-8")+".jpg" for x in cat_scenes[scene]]
            gr = create_scene_graph(annotations, im_names_all, actions, goal_im_names=goal_ims)
            scene_dict[scene.decode("utf-8")] = gr
        lbl = cat_dict[c] # use the lbl instead of the name to store in the dict
        graphs_dict[lbl] = scene_dict    
    return graphs_dict


def candidate_targets(scene, cat_dict, targets_data):
    # Find the targets that exist in the particular scene
    cats = cat_dict.keys()
    candidates=[]
    for c in cats:
        if scene.encode() in targets_data[c.encode()].keys():
            candidates.append(c)
    return candidates


def get_image_poses(world_poses, directions, im_names_all, im_names, scale):
    # Retrieve the pose (x,z,dir) for a set of images
    poses = np.zeros((len(im_names),3), dtype=np.float32)
    for i in range(len(im_names)):
        im_idx = np.where(im_names_all==im_names[i])[0]
        pos_tmp = world_poses[im_idx][0] * scale # 3 x 1
        pose_x_gt = pos_tmp[0,:]
        pose_z_gt = pos_tmp[2,:]
        dir_tmp = directions[im_idx][0] # 3 x 1
        dir_gt = np.arctan2(dir_tmp[2,:], dir_tmp[0,:])[0] # [-pi,pi], assumes that the 0 direction is to the right
        poses[i,:] = np.asarray([pose_x_gt, pose_z_gt, dir_gt], dtype=np.float32)
    return poses


def get_im_pose(im_names_all, im_name, world_poses, directions, scale):
    # Retrieve pose for a single image
    im_idx = np.where(im_names_all==im_name)[0]
    pos_tmp = world_poses[im_idx][0] * scale # 3 x 1
    pose_x_gt = pos_tmp[0,:]
    pose_z_gt = pos_tmp[2,:]
    dir_tmp = directions[im_idx][0] # 3 x 1
    dir_gt = np.arctan2(dir_tmp[2,:], dir_tmp[0,:])[0] # [-pi,pi]
    return [pose_x_gt, pose_z_gt, dir_gt]


def get_state_action_cost(current_im, actions, annotations, graph):
    # Return the costs of the actions from a certain image
    # The costs provide the supervision for imitation learning
    current_im_cost = len(nx.shortest_path(graph, current_im, "goal"))-2
    act_cost = []
    for act in actions:
        next_im = annotations[current_im][act]
        if next_im=='':
            cost = 1 # collision cost
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


def get_sseg(im_name, scene_sseg, cropSize):
    # Get sseg of a single image
    im_name = im_name.split('.')[0]
    im_sseg = scene_sseg[im_name.encode()]
    im_sseg = cv2.resize(im_sseg, (cropSize[0],cropSize[1]), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(im_sseg, axis=0)
        

def getImageData(datapath, im_name, scene, cropSize, orig_res, pixFormat, normalize, get3d=True):
    # Get necessary frame information
    scene_path = datapath + scene + "/"
    im = cv2.imread(scene_path + "/jpg_rgb/" + im_name)
    resImg = cv2.resize(im,(cropSize[0],cropSize[1]),interpolation=cv2.INTER_CUBIC)
    imgData = convert_image_by_pixformat_normalize(resImg,pixFormat,normalize)
    if get3d:
        # load and resize the depth
        depth = load_depth_file(scene_path, frame_id=im_name.split('.')[0])
        resDepth = cv2.resize(depth,(cropSize[0],cropSize[1]),interpolation=cv2.INTER_NEAREST)
        # for each location get the 3D camera coordinates x,y,z
        points2D, local3D = depth_to_3D(resDepth, getCamera(datapath, scene), orig_res, cropSize)
        return imgData, points2D, local3D
    else:
        return imgData    


def getCamera(datapath, scene):
    # Get camera information for AVD scene
    if os.path.isfile(datapath + scene + "/cameras.txt"):
        f = open(datapath + scene + "/cameras.txt")
    else: # if the scene does not have camera file then just use the one from scene 001_1
        f = open(datapath + "Home_001_1/cameras.txt")
    data = f.readlines()
    tok = data[-1].split(" ")
    intr = []
    for i in range(4,len(tok)):
        intr.append(float(tok[i]))
    return np.asarray(intr, dtype=np.float32) # fx, fy, cx, cy, distortion params


def read_label_map(label_map_path, label_to_index_path):
    # Label map for COCO detections
    f = open(label_map_path, 'r')
    data = f.readlines()
    f.close()
    labels_to_cats = {}
    for line in data:
        tok = line.split(" ")
        if len(tok)>2:
            cat = tok[1] + " " + tok[2][:-1]
        else:
            cat = tok[1][:-1]
        lbl = int(tok[0][:-1])
        labels_to_cats[lbl] = cat
    f = open(label_to_index_path, 'r') # read the file that has the correspondece of labels to indices
    data=f.readlines()
    f.close()
    labels_to_index = {}
    for line in data:
        tok = line.split(":")
        labels_to_index[int(tok[0])] = int(tok[1][:-1])
    return labels_to_cats, labels_to_index


def generate_detection_image(detections, image_size, num_classes, labels_to_index, is_binary):
    # Creates a frame with the detection masks
    res = np.zeros((num_classes, image_size[1], image_size[0]), dtype=np.float32)
    boxes = detections[b'detection_boxes']
    labels = detections[b'detection_classes']
    scores = detections[b'detection_scores']
    nDets = detections[b'num_detections']
    for i in range(nDets):
        lbl = labels[i]
        score = scores[i]
        box=boxes[i] # top left bottom right
        y1 = int(box[0]*image_size[1])
        x1 = int(box[1]*image_size[0])
        y2 = int(box[2]*image_size[1])
        x2 = int(box[3]*image_size[0])
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


def load_detections(par, scene_list):
    labels_to_cats, labels_to_index = read_label_map(par.label_map_path, par.label_index_path)
    detection_data = {}
    for sc in scene_list:
        det_path = par.det_dir_path + sc + ".npy"
        scene_dets = np.load(det_path, encoding='bytes', allow_pickle=True).item()
        detection_data[sc] = scene_dets
    return detection_data, labels_to_cats, labels_to_index


def get_det_mask(im_name, scene_dets, cropSize, dets_nClasses, labels_to_index):
    im_name = im_name.split('.')[0]
    im_dets = scene_dets[im_name.encode()]
    det_mask = generate_detection_image(detections=im_dets, image_size=cropSize, 
                                            num_classes=dets_nClasses, labels_to_index=labels_to_index, is_binary=True)
    return det_mask
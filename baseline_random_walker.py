import collections
import copy
import json
import os
import time
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from math import cos, sin, acos, atan2, pi
from io import StringIO
import png
from statistics import mean
from baseline_utils import target_category_list, mapper_cat2index, TRAIN_WORLDS, TEST_WORLDS, SUPPORTED_ACTIONS, minus_theta_fn, cameraPose2currentPose, readDepthImage, project_pixels_to_world_coords, read_all_poses, read_cached_data, ActiveVisionDatasetEnv

# setup parameters
dataset_dir = '/home/reza/Datasets/ActiveVisionDataset/AVD_Minimal'
saved_folder = 'baseline_random_train_temp'
detection_thresh = 0.9
mode = 'train' #'test'

#=======================================================================================================================
np.set_printoptions(precision=2, suppress=True)
np.random.seed(0)

if not os.path.exists(saved_folder):
  os.mkdir(saved_folder)

#=======================================================================================================================
if mode == 'train':
  WORLDS = TRAIN_WORLDS
elif mode == 'test':
  WORLDS = TEST_WORLDS

for world_id in range(len(WORLDS)):
  current_world = WORLDS[world_id]
  dataset_root = dataset_dir

  ## key: img_name, val: (x, z, rot, scale)
  all_poses = read_all_poses(dataset_root, current_world)
  cached_data = read_cached_data(True, dataset_root, targets_file_name=None, output_size=224, Home_name=current_world.encode()) ## encode() convert string to byte
  all_init = np.load('{}/Meta/all_init_configs.npy'.format(dataset_root), allow_pickle=True).item()
  ## collect init img ids for current world
  list_init_img_id = []
  for pair in all_init[current_world.encode()]:
    init_img_id, _ = pair
    init_img_id = init_img_id.decode()
    if init_img_id not in list_init_img_id:
      list_init_img_id.append(init_img_id)
  annotated_targets = np.load('{}/Meta/annotated_targets.npy'.format(dataset_root), allow_pickle=True).item()
  detections = np.load('{}/Meta/Detections/{}.npy'.format(dataset_root, current_world), encoding='bytes', allow_pickle=True).item()
  ## list of image ids 
  ## for example, current_world_image_ids[0].decode
  current_world_image_ids = cached_data['world_id_dict'][current_world.encode()]
  ## initialize the graph map
  AVD = ActiveVisionDatasetEnv(current_world_image_ids, current_world, dataset_root)
  ## load true thetas
  scene_path = '{}/{}'.format(dataset_dir, current_world)
  image_structs_path = os.path.join(scene_path,'image_structs.mat')
  image_structs = sio.loadmat(image_structs_path)
  image_structs = image_structs['image_structs']
  image_structs = image_structs[0]

  ##============================================================================================================
  ## go through each target_category
  for target_category in target_category_list:
    ## check if current_world has the target_category
    if current_world in annotated_targets[target_category].keys():
      print('target_category {} in current_world {}'.format(target_category, current_world))
      sum_success = 0
      list_ratio_optimal_policy = []
      category_index = mapper_cat2index[target_category]

      ## compute target_views for current_category in current_world
      annotated_img_id = annotated_targets[target_category][current_world]
      filtered_img_id = []
      for img_id in annotated_img_id:
        if img_id != '':
          filtered_img_id.append(img_id)
      annotated_img_id = filtered_img_id

      for idx, init_img_id in enumerate(list_init_img_id):
        if idx >= 10:
          break

        current_img_id = init_img_id
        ## keep record of the visited imgs and actions
        list_visited_img_id = [current_img_id]
        list_actions = []

  ## step 1: rotate to look for the target and move randomly to neighboring vertex clusters
        ## repeat until see the target category
        flag_target_detected = False
        while True:
          ## check if current image contains the target category
          current_detection = detections[current_img_id.encode()]
          if len(np.where(current_detection[b'detection_classes'] == category_index)[0]) > 0:
            detection_id = np.where(current_detection[b'detection_classes'] == category_index)[0][0]
            detection_bbox = current_detection[b'detection_boxes'][detection_id]
            y1, x1, y2, x2 = [int(round(t)) for t in detection_bbox * 224]
            detection_score = current_detection[b'detection_scores'][detection_id]
            if (y2 - y1) * (x2 - x1) > 0 and detection_score > detection_thresh:
              flag_target_detected = True

          ## check if we detect the category when rotate the viewpoint previously
          if flag_target_detected: ## get out the top while loop
            break

          if len(list_actions) > 100:
            break

          ## move to another vertex cluster
          while True:
            action = np.random.choice(SUPPORTED_ACTIONS)
            if action != 'stop':
              next_img_id = AVD._next_image(current_img_id, action)
              if next_img_id != '':
                break
          list_visited_img_id.append(next_img_id)
          list_actions.append(action)
          current_img_id = next_img_id

  ## step 2: localize the point closest to target category point cloud and plan the path
        ## find the bbox
        current_detection = detections[current_img_id.encode()]
        if len(np.where(current_detection[b'detection_classes'] == category_index)[0]) > 0:
          detection_id = np.where(current_detection[b'detection_classes'] == category_index)[0][0]
          detection_bbox = current_detection[b'detection_boxes'][detection_id]
          y1, x1, y2, x2 = [int(round(t)) for t in detection_bbox * 224]

        fig, ax = plt.subplots(1)
        current_img = cached_data['IMAGE'][current_img_id.encode()]
        ax.imshow(current_img)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=5,edgecolor='green',facecolor=(0,1,0,0.5))
        ax.add_patch(rect)
        plt.title('env: {}, target: {}, detection_score: {:.2f}'.format(current_world, target_category, detection_score))
        plt.savefig('{}/env_{}_category_{}_id_{}_left.jpg'.format(saved_folder, current_world, target_category, idx), bbox_inches='tight')
        plt.close()

        ## project object pixels and find the points (x, z)
        current_depth = readDepthImage(current_world, current_img_id, dataset_dir)
        current_camera_pose = all_poses[current_img_id] ## x, z, R, f
        current_pose, direction = cameraPose2currentPose(current_img_id, current_camera_pose, image_structs)
        middle_img_id = current_img_id
        object_points_2d = project_pixels_to_world_coords(current_depth, current_pose, [y1, x1, y2, x2])
        ## localize the target pose into a target_img
        #print('localize the target pose among all the world imgs ...')
        dist_to_object_points = np.zeros(len(current_world_image_ids), dtype=np.float32)
        for j, image_id in enumerate(current_world_image_ids):
          x, z, _, _ = all_poses[image_id.decode()]
          image_point = np.array([[x], [z]]) ## shape: 2 x 1
          diff_object_points = np.repeat(image_point, object_points_2d.shape[1], axis=1) - object_points_2d
          dist_to_object_points[j] =  np.sum((diff_object_points**2).flatten())
        argmin_dist_to_object_points = np.argmin(dist_to_object_points)
        target_img_id = current_world_image_ids[argmin_dist_to_object_points].decode()

        ## compute shortest path to target_img_id from current_id
        current_img_vertex = AVD.to_vertex(current_img_id)
        target_img_vertex = AVD.to_vertex(target_img_id)
        path = nx.shortest_path(AVD._cur_graph.graph, current_img_vertex, target_img_vertex)
        ## add intermediate points and actions to list_visited_img_id
        ## omit the start vertex since it's already included in list_visited_img_id
        for j in range(1, len(path)):
          img_id = AVD.to_image_id(path[j])
          list_visited_img_id.append(img_id)
        for j in range(len(path)-1):
          list_actions.append(AVD.action(path[j], path[j + 1]))
        current_img_id = target_img_id

  ## step 3: rotate and find the best view towards the target
        ## look around to see if target category is there
        flag_target_detected = False
        list_look_around_img_id = [] ## I don't know how many times to rotate, so keep the record of it
        while current_img_id not in list_look_around_img_id:
          ## check if the image contains the target category
          current_detection = detections[current_img_id.encode()]
          if len(np.where(current_detection[b'detection_classes'] == category_index)[0]) > 0:
            detection_id = np.where(current_detection[b'detection_classes'] == category_index)[0][0]
            detection_bbox = current_detection[b'detection_boxes'][detection_id]
            y1, x1, y2, x2 = [int(round(t)) for t in detection_bbox * 224]
            detection_score = current_detection[b'detection_scores'][detection_id]
            if (y2 - y1) * (x2 - x1) > 0 and detection_score > detection_thresh:
              flag_target_detected = True
              break ## get out of the inner while loop

          list_look_around_img_id.append(current_img_id)
          ## rotate_cw
          action = 'rotate_cw'
          next_img_id = AVD._next_image(current_img_id, action)

          list_visited_img_id.append(next_img_id)
          list_actions.append(action)
          current_img_id = next_img_id

  ## Evaluation stage
        num_steps = len(list_actions)
        ## compute steps to one of the annotated views
        steps_to_annotated_imgs = np.zeros(len(annotated_img_id), dtype=np.int16)
        for j, target_img_id in enumerate(annotated_img_id):
          current_img_vertex = AVD.to_vertex(current_img_id)
          target_img_vertex = AVD.to_vertex(target_img_id)
          path = nx.shortest_path(AVD._cur_graph.graph, current_img_vertex, target_img_vertex)
          steps_to_annotated_imgs[j] = len(path)-1
        minimum_steps = min(steps_to_annotated_imgs)
        if minimum_steps <= 5:
          success = True
          sum_success += 1
        else:
          success = False

        ## compute optimal path from init_point to target_point
        optimal_steps_to_annotated_imgs = np.ones(len(annotated_img_id), dtype=np.int16)
        for j, target_img_id in enumerate(annotated_img_id):
          init_img_vertex = AVD.to_vertex(init_img_id)
          target_img_vertex = AVD.to_vertex(target_img_id)
          path = nx.shortest_path(AVD._cur_graph.graph, init_img_vertex, target_img_vertex)
          optimal_steps_to_annotated_imgs[j] = len(path)-1
        minimum_optimal_steps = min(optimal_steps_to_annotated_imgs)
        if minimum_optimal_steps == 0:
          minimum_optimal_steps = 1.0
        ratio_optimal_policy = 1.0 * num_steps / minimum_optimal_steps
        if success:
          list_ratio_optimal_policy.append(ratio_optimal_policy)

  ## ==========================================================================================================
        ## draw the trajectory
        ##  draw all the points in the world
        for key, val in all_poses.items():
          x, z, rot, scale = val
          plt.plot(x, z, color='blue', marker='o', markersize=5)
        ##  draw the projected target category points
        for i in range(object_points_2d.shape[1]):
          plt.plot(object_points_2d[0, i], object_points_2d[1, i], color='violet', marker='o', markersize=5)
        ## draw the annotated views
        for img_id in annotated_img_id:
          x, z, rot, scale = all_poses[img_id]
          plt.plot(x, z, color='yellow', marker='v', markersize=10)
        ##  draw the path:
        xs = []
        zs = []
        for img_id in list_visited_img_id:
          x, z, rot, scale = all_poses[img_id]
          xs.append(x)
          zs.append(z)
        plt.plot(xs, zs, color='black', marker='o', markersize=5)
        ## draw the start point and end point and middle point
        x, z, rot, scale = all_poses[list_visited_img_id[0]]
        plt.plot(x, z, color='green', marker='.', markersize=10)
        x, z, rot, scale = all_poses[middle_img_id]
        plt.plot(x, z, color='cyan', marker='.', markersize=10)
        x, z, rot, scale = all_poses[list_visited_img_id[-1]]
        plt.plot(x, z, color='red', marker='.', markersize=10)
        ## draw arrow
        x, z, rot, scale = all_poses[middle_img_id]
        theta = atan2(direction[2], direction[0])
        #theta = minus_theta_fn(theta, pi/2)
        end_x = x + cos(theta)
        end_z = z + sin(theta)
        plt.arrow(x, z, 2*cos(theta), 2*sin(theta), head_width=0.3, head_length=0.4, fc='r', ec='r')
        plt.grid()
        plt.title('env: {}, target: {}, steps: {}, success: {}\noptimal steps: {}, ratio: {:.2f}'.format(current_world, target_category, num_steps, success, minimum_optimal_steps, ratio_optimal_policy))
        #plt.show()
        plt.axis('scaled')
        if world_id == 0:
          plt.xticks(np.arange(-6, 5, 1.0))
          plt.yticks(np.arange(-8, 6, 1.0))
        elif world_id == 1:
          plt.xticks(np.arange(-8, 5, 1.0))
          plt.yticks(np.arange(-5, 4, 1.0))
        else:
          plt.xticks(np.arange(-6, 9, 1.0))
          plt.yticks(np.arange(-5, 5, 1.0))
        plt.savefig('{}/env_{}_category_{}_id_{}_middle.jpg'.format(saved_folder, current_world, target_category, idx), bbox_inches='tight')
        plt.close()
        
        ## draw the bbox
        fig, ax = plt.subplots(1)
        current_img = cached_data['IMAGE'][current_img_id.encode()]
        ax.imshow(current_img)

        current_detection = detections[current_img_id.encode()]
        detection_score = 0.0
        if len(np.where(current_detection[b'detection_classes'] == category_index)[0]) > 0:
          detection_id = np.where(current_detection[b'detection_classes'] == category_index)[0][0]
          detection_bbox = current_detection[b'detection_boxes'][detection_id]
          y1, x1, y2, x2 = [int(round(t)) for t in detection_bbox * 224]
          detection_score = current_detection[b'detection_scores'][detection_id]
          rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=5, edgecolor='green', facecolor=(0,1,0,0.5))
          ax.add_patch(rect)
        plt.title('env: {}, target: {}, detection_score: {:.2f}'.format(current_world, target_category, detection_score))
        plt.savefig('{}/env_{}_category_{}_id_{}_right.jpg'.format(saved_folder, current_world, target_category, idx), bbox_inches='tight')
        plt.close()

      ## compute average ratio of the successful runs
      avg_ratio = mean(list_ratio_optimal_policy)
      print('env: {}, category: {}, success rate: {}/{}, avg_ratio: {:.2f}'.format(current_world, target_category, sum_success, len(list_init_img_id), avg_ratio))

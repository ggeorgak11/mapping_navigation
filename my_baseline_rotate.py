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

AVD_dir = '/Users/yimengli/work/cognitive_planning_original/AVD_Minimal'
saved_folder = '/Users/yimengli/work/cognitive_planning_original/baseline_rotate'

np.set_printoptions(precision=2, suppress=True)
np.random.seed(0)

TEST_WORLDS = ['Home_011_1', 'Home_013_1', 'Home_016_1']
SUPPORTED_ACTIONS = ['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']
_Graph = collections.namedtuple('_Graph', ['graph', 'id_to_index', 'index_to_id'])
detection_thresh = 0.9

def minus_theta_fn(previous_theta, current_theta):
  result = current_theta - previous_theta
  if result < -math.pi:
    result += 2*math.pi
  if result > math.pi:
    result -= 2*math.pi
  return result

def cameraPose2currentPose (current_img_id, camera_pose):
  current_x = camera_pose[0]
  current_z = camera_pose[1]
  for i in range(image_structs.shape[0]):
    if image_structs[i][0].item()[:-4] == current_img_id:
      direction = image_structs[i][4]
      break
  current_theta = atan2(direction[2], direction[0])
  current_theta = minus_theta_fn(current_theta, pi/2)
  #print('current_theta = {}'.format(current_theta))
  current_pose = [current_x, current_z, current_theta]
  return current_pose, direction

def readDepthImage (current_world, current_img_id, resolution=224):
  img_id = current_img_id[:-1]+'3'
  reader = png.Reader('{}/{}/high_res_depth/{}.png'.format(AVD_dir, current_world, img_id))
  data = reader.asDirect()
  pixels = data[2]
  image = []
  for row in pixels:
    row = np.asarray(row)
    image.append(row)
  image = np.stack(image, axis=0)
  image = image.astype(np.float32)
  image = image/1000.0
  depth = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
  return depth

def project_pixels_to_world_coords (current_depth, current_pose, bbox, gap=2, focal_length=112, resolution=224, start_pixel=1):
  def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
    x1, y1, theta1 = next_pose
    x0, y0, theta0 = current_pose
    phi = math.atan2(y1-y0, x1-x0)
    gamma = minus_theta_fn(theta0, phi)
    dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    tz = dist * math.cos(gamma)
    tx = -dist * math.sin(gamma)
    theta_change = (theta1 - theta0)
    #print('dist = {}'.format(dist))
    #print('gamma = {}'.format(gamma))
    #print('theta_change = {}'.format(theta_change))
    return tx, tz, theta_change

  y1, x1, y2, x2 = bbox
  x = [i for i in range(x1+start_pixel, x2-start_pixel, gap)]
  y = [i for i in range(y1+start_pixel, y2-start_pixel, gap)]
  ## densely sample keypoints for current image
  ## first axis of kp1 is 'u', second dimension is 'v'
  kp1 = np.empty((2, len(x)*len(y)))
  count = 0
  for i in range(len(x)):
    for j in range(len(y)):
      kp1[0, count] = x[i]
      kp1[1, count] = y[j]
      count += 1
  ## camera intrinsic matrix
  K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
  #print('K = {}'.format(K))
  ## expand kp1 from 2 dimensions to 3 dimensions
  kp1_3d = np.ones((3, kp1.shape[1]))
  kp1_3d[:2, :] = kp1

  ## backproject kp1_3d through inverse of K and get kp1_3d. x=KX, X is in the camera frame
  ## Now kp1_3d still have the third dimension Z to be 1.0. This is the world coordinates in camera frame after projection.
  kp1_3d = LA.inv(K).dot(kp1_3d)
  #print('kp1_3d: {}'.format(kp1_3d))
  ## backproject kp1_3d into world coords kp1_4d by using gt-depth
  ## Now kp1_4d has coords in world frame if camera1 (current) frame coincide with the world frame
  kp1_4d = np.ones((4, kp1.shape[1]))
  good = []
  for i in range(kp1.shape[1]):
    Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
    #print('Z = {}'.format(Z))
    kp1_4d[2, i] = Z
    kp1_4d[0, i] = Z * kp1_3d[0, i]
    kp1_4d[1, i] = Z * kp1_3d[1, i]
    #Z_mask = current_depth[int(kp1[1, i]), int(kp1[0, i]), 1]
    if Z > 0:
      good.append(i)
  kp1_4d = kp1_4d[:, good]
  #print('kp1_4d: {}'.format(kp1_4d))
  
  ## first compute the rotation and translation from current frame to goal frame
  '''
  goal_pose = [0.0, 0.0, 0.0]
  tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)
  #print('tx={}, tz={}, theta={}'.format(tx, tz, theta))
  R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  '''
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  #transformation_matrix[:3, :3] = R.T
  #transformation_matrix[:3, 3] = -R.T.dot(T)

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  kp2_3d = transformation_matrix.dot(kp1_4d)
  ## pick x-row and z-row
  kp2_2d = kp2_3d[[0, 2], :]

  return kp2_2d

def read_all_poses(dataset_root, world):
  """Reads all the poses for each world.

  Args:
    dataset_root: the path to the root of the dataset.
    world: string, name of the world.

  Returns:
    Dictionary of poses for all the images in each world. The key is the image
    id of each view and the values are tuple of (x, z, R, scale). Where x and z
    are the first and third coordinate of translation. R is the 3x3 rotation
    matrix and scale is a float scalar that indicates the scale that needs to
    be multipled to x and z in order to get the real world coordinates.

  Raises:
    ValueError: if the number of images do not match the number of poses read.
  """
  path = os.path.join(dataset_root, world, 'image_structs.mat')
  data = sio.loadmat(path)

  xyz = data['image_structs']['world_pos']
  image_names = data['image_structs']['image_name'][0]
  rot = data['image_structs']['R'][0]
  scale = data['scale'][0][0]
  n = xyz.shape[1]
  x = [xyz[0][i][0][0] for i in range(n)]
  z = [xyz[0][i][2][0] for i in range(n)]
  names = [name[0][:-4] for name in image_names]
  if len(names) != len(x):
    raise ValueError('number of image names are not equal to the number of '
                     'poses {} != {}'.format(len(names), len(x)))
  output = {}
  for i in range(n):
    if rot[i].shape[0] != 0:
      assert rot[i].shape[0] == 3
      assert rot[i].shape[1] == 3
      output[names[i]] = (x[i], z[i], rot[i], scale)
    else:
      output[names[i]] = (x[i], z[i], None, scale)

  return output

def read_cached_data(should_load_images, dataset_root, targets_file_name, output_size, Home_name):
  """Reads all the necessary cached data.

  Args:
    should_load_images: whether to load the images or not.
    dataset_root: path to the root of the dataset.
    segmentation_file_name: The name of the file that contains semantic
      segmentation annotations.
    targets_file_name: The name of the file the contains targets annotated for
      each world.
    output_size: Size of the output images. This is used for pre-processing the
      loaded images.
  Returns:
    Dictionary of all the cached data.
  """

  result_data = {}
  
  ## loading targets
  #annotated_target_path = os.path.join(dataset_root, 'Meta', targets_file_name + '.npy')
  #result_data['targets'] = np.load(annotated_target_path).item()
  '''
  depth_image_path = os.path.join(dataset_root, 'Meta/depth_imgs.npy')
  ## loading depth
  depth_data = np.load(depth_image_path, encoding='bytes').item()

  ## processing depth
  for home_id in depth_data:
    if home_id == Home_name:
      images = depth_data[home_id]
      for image_id in images:
        depth = images[image_id]
        assert 1==2
        depth = cv2.resize(
            depth / _MAX_DEPTH_VALUE, (output_size, output_size),
            interpolation=cv2.INTER_NEAREST)
        depth_mask = (depth > 0).astype(np.float32)
        depth = np.dstack((depth, depth_mask))
        images[image_id] = depth

  result_data['DEPTH'] = depth_data[Home_name]
  '''
  if should_load_images:
    image_path = os.path.join(dataset_root, 'Meta/imgs.npy')
    ## loading imgs
    image_data = np.load(image_path, encoding='bytes').item()
    result_data['IMAGE'] = image_data[Home_name]

  word_id_dict_path = os.path.join(dataset_root, 'Meta/world_id_dict.npy')
  result_data['world_id_dict'] = np.load(word_id_dict_path, encoding='bytes').item()

  return result_data

##==========================================================================================================================================================================
class ActiveVisionDatasetEnv():
  def __init__(self, image_list, current_world, dataset_root):
    self._episode_length = 50
    self._cur_graph = None  # Loaded by _update_graph
    self._world_image_list = image_list
    self._actions = SUPPORTED_ACTIONS
    ## load json file
    f = open('{}/{}/annotations.json'.format(dataset_root, current_world))
    file_content = f.read()
    file_content = file_content.replace('.jpg', '')
    io = StringIO(file_content)
    self._all_graph = json.load(io)
    f.close()

    self._update_graph()

  def to_image_id(self, vid):
    """Converts vertex id to the image id.

    Args:
      vid: vertex id of the view.
    Returns:
      image id of the input vertex id.
    """
    return self._cur_graph.index_to_id[vid]

  def to_vertex(self, image_id):
    return self._cur_graph.id_to_index[image_id]

  def _next_image(self, image_id, action):
    """Given the action, returns the name of the image that agent ends up in.
    Args:
      image_id: The image id of the current view.
      action: valid actions are ['right', 'rotate_cw', 'rotate_ccw',
      'forward', 'left']. Each rotation is 30 degrees.

    Returns:
      The image name for the next location of the agent. If the action results
      in collision or it is not possible for the agent to execute that action,
      returns empty string.
    """
    return self._all_graph[image_id][action]

  def action(self, from_index, to_index):
    return self._cur_graph.graph[from_index][to_index]['action']

  def _update_graph(self):
    """Creates the graph for each environment and updates the _cur_graph."""
    graph = nx.DiGraph()
    id_to_index = {}
    index_to_id = {}
    image_list = self._world_image_list
    for i, image_id in enumerate(image_list):
      image_id = image_id.decode()
      id_to_index[image_id] = i
      index_to_id[i] = image_id
      graph.add_node(i)

    for image_id in image_list:
      image_id = image_id.decode()
      for action in self._actions:
        if action == 'stop':
          continue
        next_image = self._all_graph[image_id][action]
        if next_image:
          graph.add_edge(id_to_index[image_id], id_to_index[next_image], action=action)
    self._cur_graph = _Graph(graph, id_to_index, index_to_id)
##==========================================================================================================
for world_id in range(len(TEST_WORLDS)):
  #world_id = 0
  current_world = TEST_WORLDS[world_id]
  dataset_root = AVD_dir
  _MAX_DEPTH_VALUE = 12102
  target_category_list = ['tv', 'dining_table', 'fridge', 'microwave', 'couch']
  mapper_cat2index = {'tv': 72, 'dining_table': 67, 'fridge': 82, 'microwave': 78, 'couch': 63}
  #category_id = 2
  #target_category = target_category_list[category_id]
  #category_index = mapper_cat2index[target_category]

  ## key: img_name, val: (x, z, rot, scale)
  all_poses = read_all_poses(dataset_root, current_world)
  ## cached_data['DEPTH'][b'img_id'] or cached_data['IMAGE'][b'img_id']
  cached_data = read_cached_data(True, dataset_root, targets_file_name=None, output_size=224, Home_name=current_world.encode()) ## encode() convert string to byte
  ## all_init[b'Home_id']: list of (b'init_image', b'target_category')
  all_init = np.load('{}/Meta/all_init_configs.npy'.format(dataset_root)).item()
  ## collect init img ids for current world
  list_init_img_id = []
  for pair in all_init[current_world.encode()]:
    init_img_id, _ = pair
    init_img_id = init_img_id.decode()
    if init_img_id not in list_init_img_id:
      list_init_img_id.append(init_img_id)
  ## annotated_targets[category][Home_id], categories include 'tv' 72, 'dining_table' 67, 'fridge' 82, 'microwave' 78, 'couch' 63
  annotated_targets = np.load('{}/Meta/annotated_targets.npy'.format(dataset_root)).item()
  ## detections[b'img_id'] include [b'detection_scores'], [b'detection_boxes'], [b'num_detections'], [b'detection_classes']
  detections = np.load('{}/Meta/Detections/{}.npy'.format(dataset_root, current_world), encoding='bytes').item()
  ## list of image ids 
  ## for example, current_world_image_ids[0].decode
  current_world_image_ids = cached_data['world_id_dict'][current_world.encode()]
  ## initialize the graph map
  AVD = ActiveVisionDatasetEnv(current_world_image_ids, current_world, dataset_root)
  ## load true thetas
  scene_path = '{}/{}'.format(AVD_dir, current_world)
  image_structs_path = os.path.join(scene_path,'image_structs.mat')
  image_structs = sio.loadmat(image_structs_path)
  image_structs = image_structs['image_structs']
  image_structs = image_structs[0]
  #print('Finished loading the world: {}'.format(current_world))

  ##============================================================================================================
  ## go through each target_category
  for target_category in target_category_list:
    ## check if current_world has the target_category
    if current_world in annotated_targets[target_category].keys():
      #print('target_category {} in current_world {}'.format(target_category, current_world))
      sum_success = 0
      list_ratio_optimal_policy = []
      category_index = mapper_cat2index[target_category]

      ## compute target_views for current_category in current_world
      annotated_img_id = annotated_targets[target_category][current_world]

      for idx, init_img_id in enumerate(list_init_img_id):
        #print('init_img_id: {}'.format(init_img_id))
        current_img_id = init_img_id
        ## keep record of the visited imgs and actions
        list_visited_img_id = [current_img_id]
        list_actions = []

  ## step 1: rotate to look for the target and move randomly to neighboring vertex clusters
        ## repeat until see the target category
        flag_target_detected = False
        while True:
          ## look around to see if target category is there
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
                #assert 1==2
                flag_target_detected = True
                break ## get out of the inner while loop

            list_look_around_img_id.append(current_img_id)
            ## rotate_cw
            action = 'rotate_cw'
            next_img_id = AVD._next_image(current_img_id, action)

            list_visited_img_id.append(next_img_id)
            list_actions.append(action)
            current_img_id = next_img_id

          ## check if we detect the category when rotate the viewpoint previously
          if flag_target_detected: ## get out the top while loop
            break

          if len(list_actions) > 1000:
            break

          ## move to another vertex cluster
          while True:
            action = np.random.choice(SUPPORTED_ACTIONS)
            if action != 'stop' and action != 'rotate_cw' and action !='rotate_ccw':
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
        #plt.show()

        ## project object pixels and find the points (x, z)
        current_depth = readDepthImage(current_world, current_img_id)
        current_camera_pose = all_poses[current_img_id] ## x, z, R, f
        current_pose, direction = cameraPose2currentPose(current_img_id, current_camera_pose)
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
        #print('x = {}, z = {}, end_x = {}, end_z = {}'.format(x, z, end_x, end_z))
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
        #plt.show()
        plt.savefig('{}/env_{}_category_{}_id_{}_right.jpg'.format(saved_folder, current_world, target_category, idx), bbox_inches='tight')
        plt.close()

      ## compute average ratio of the successful runs
      avg_ratio = mean(list_ratio_optimal_policy)
      print('env: {}, category: {}, success rate: {}/{}, avg_ratio: {:.2f}'.format(current_world, target_category, sum_success, len(list_init_img_id), avg_ratio))


'''
## find an image contains the target category
current_img_id = annotated_targets[target_category][current_world][2]
current_img = cached_data['IMAGE'][current_img_id.encode()]
current_depth = cached_data['DEPTH'][current_img_id.encode()]
current_detection = detections[current_img_id.encode()]
current_camera_pose = all_poses[current_img_id] ## x, z, R, f
current_pose = cameraPose2currentPose(current_camera_pose)

## find the bbox
detection_id = np.where(current_detection[b'detection_classes'] == category_index)[0][0]
detection_bbox = current_detection[b'detection_boxes'][detection_id]
y1, x1, y2, x2 = [int(round(t)) for t in detection_bbox * 224]

## project object pixels
object_points_2d = project_pixels_to_world_coords(current_depth, current_pose, [y1, x1, y2, x2])
'''







'''
## draw poses
xs = []
zs = []
for key, val in all_poses.items():
  x, z, rot, scale = val
  xs.append(x)
  zs.append(z)
for i in range(len(xs)):
  plt.plot(xs[i], zs[i], color='blue', marker='o', markersize=1)
for i in range(object_points_2d.shape[1]):
  plt.plot(object_points_2d[0, i], object_points_2d[1, i], color='red', marker='o', markersize=1)
plt.grid()
plt.title('{}'.format(current_world))
plt.show()
'''

'''
## draw the bbox
fig, ax = plt.subplots(1)
ax.imshow(current_img)
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()
'''

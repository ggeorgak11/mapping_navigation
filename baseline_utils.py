import collections
import copy
import json
import os
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2
import math
from math import cos, sin, acos, atan2, pi
from io import StringIO
import png

TRAIN_WORLDS = ['Home_002_1', 'Home_005_1']
TEST_WORLDS = ['Home_011_1', 'Home_013_1', 'Home_016_1']

SUPPORTED_ACTIONS = ['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']

_Graph = collections.namedtuple('_Graph', ['graph', 'id_to_index', 'index_to_id'])

target_category_list = ['tv', 'dining_table', 'fridge', 'microwave', 'couch']
mapper_cat2index = {'tv': 72, 'dining_table': 67, 'fridge': 82, 'microwave': 78, 'couch': 63}

def minus_theta_fn(previous_theta, current_theta):
  result = current_theta - previous_theta
  if result < -math.pi:
    result += 2*math.pi
  if result > math.pi:
    result -= 2*math.pi
  return result

def cameraPose2currentPose (current_img_id, camera_pose, image_structs):
  current_x = camera_pose[0]
  current_z = camera_pose[1]
  for i in range(image_structs.shape[0]):
    if image_structs[i][0].item()[:-4] == current_img_id:
      direction = image_structs[i][4]
      break
  current_theta = atan2(direction[2], direction[0])
  current_theta = minus_theta_fn(current_theta, pi/2)
  current_pose = [current_x, current_z, current_theta]
  return current_pose, direction

def readDepthImage (current_world, current_img_id, AVD_dir, resolution=224):
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
  ## expand kp1 from 2 dimensions to 3 dimensions
  kp1_3d = np.ones((3, kp1.shape[1]))
  kp1_3d[:2, :] = kp1

  ## backproject kp1_3d through inverse of K and get kp1_3d. x=KX, X is in the camera frame
  ## Now kp1_3d still have the third dimension Z to be 1.0. This is the world coordinates in camera frame after projection.
  kp1_3d = LA.inv(K).dot(kp1_3d)
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
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T

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
  
  if should_load_images:
    image_path = os.path.join(dataset_root, 'Meta/imgs.npy')
    ## loading imgs
    image_data = np.load(image_path, encoding='bytes', allow_pickle=True).item()
    result_data['IMAGE'] = image_data[Home_name]

  word_id_dict_path = os.path.join(dataset_root, 'Meta/world_id_dict.npy')
  result_data['world_id_dict'] = np.load(word_id_dict_path, encoding='bytes', allow_pickle=True).item()

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

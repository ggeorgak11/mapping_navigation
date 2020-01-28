import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as plt
from plyfile import PlyData
import data_helper as dh
import pickle
from absl import flags

# This script generates visualizations of either navigation or localization episodes.
# For navigation each step shows the map with the current agent position, and the egocentric image.
# For localization each step shows the map with the ground-truth (blue) and predicted (red) trajectories.

# Input arguments 
flags.DEFINE_string("avd_root", "", "Path to AVD data.")
flags.DEFINE_string("episode_results_file", "", "File saved by the test_NavNet.py script.")
flags.DEFINE_string("scene", "", "Scene used when testing the model.")
flags.DEFINE_string("save_path", "", "Where to write the figures.")
flags.DEFINE_enum("mode", "nav", ["nav", "loc"], "Indicates whether the evaluation results are for navigation or localization")


def get_pcloud(scene_path, scale):
    data = PlyData.read(scene_path+'dense_reconstruction.ply')
    print("Loaded the 3D point cloud!")
    x, y, z = data['vertex']['x'], data['vertex']['y'], data['vertex']['z']
    r, g, b = data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']
    pcloud = np.zeros((x.shape[0], 3), dtype=np.float32)
    pcloud[:,0] = x
    pcloud[:,1] = y
    pcloud[:,2] = z
    pcloud = pcloud[::100, :]
    # keep points below the ceiling
    ceiling = 0
    valid_inds = np.where(pcloud[:,1] > ceiling)[0]
    pcloud = pcloud[valid_inds, :]
    pcloud = pcloud * scale
    color_cloud = np.zeros((r.shape[0], 3), dtype=np.int)
    color_cloud[:,0] = r
    color_cloud[:,1] = g
    color_cloud[:,2] = b
    color_cloud = color_cloud[::100, :]
    color_cloud = color_cloud[valid_inds, :]
    return pcloud, color_cloud


def get_images(imgs_name, scene_path):
    im0 = cv2.imread(scene_path + "/jpg_rgb/" + imgs_name[0])
    im_h, im_w, _ = im0.shape
    im_list = np.zeros((imgs_name.shape[0], im_h, im_w, 3), dtype=np.float32)
    im_list[0,:,:,:] = im0[:,:,::-1]
    for i in range(1,len(imgs_name)):
        if imgs_name[i]=='': # in case of collision just keep the same image as before
            im_list[i,:,:,:] = im_list[i-1,:,:,:]
        else:
            im = cv2.imread(scene_path + "/jpg_rgb/" + imgs_name[i])
            im_list[i,:,:,:] = im[:,:,::-1]
    return im_list


def plot_step_nav(ax, k, im_list, poses, save_dir=None):
    ax[0].scatter(poses[k,0], poses[k,1], color="red", s=3)
    if k>0: # draw a line between the previous and current pose
        ax[0].plot([poses[k-1,0], poses[k,0]], [poses[k-1,1], poses[k,1]], 'r-', linewidth=0.5)
    ax[1].imshow(im_list[k,:,:,:]/255.0)
    for a in ax:
        a.set_aspect('equal')
        a.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir+str(k)+".png", bbox_inches='tight', pad_inches=0, dpi=200)
    else:
        plt.show()


def plot_step_loc(ax, k, poses_pred, poses_gt, save_dir=None):
    ax.scatter(poses_gt[k,0], poses_gt[k,1], color="blue", s=12)
    ax.scatter(poses_pred[k,0], poses_pred[k,1], color="red", s=12)
    if k>0: # draw a line between the previous and current pose
        ax.plot([poses_gt[k-1,0], poses_gt[k,0]], [poses_gt[k-1,1], poses_gt[k,1]], 'b-', linewidth=0.5)
        ax.plot([poses_pred[k-1,0], poses_pred[k,0]], [poses_pred[k-1,1], poses_pred[k,1]], 'r-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    if save_dir is not None:
        plt.savefig(save_dir+str(k)+".png", bbox_inches='tight', pad_inches=0, dpi=200)
    else:
        plt.show()


def visualize_nav(avd_root, episode_results, scene, save_path):
    # Load the targets info
    targets_file_path = avd_root + "Meta/annotated_targets.npy"
    targets_data = np.load(targets_file_path, encoding='bytes', allow_pickle=True).item()
    # Load scene info
    annotations, scale, im_names_all, world_poses, directions = dh.load_scene_info(avd_root, scene)
    # Load the pcloud
    scene_path = avd_root + scene + "/" # to read the images
    pcloud, color_cloud = get_pcloud(scene_path=scene_path, scale=scale)
    for i in range(len(episode_results)):
        (image_seq, action_seq, cat, done) = episode_results[i]
        image_seq = np.asarray(image_seq)
        #print("image_seq:", image_seq)
        im_list = get_images(imgs_name=image_seq, scene_path=scene_path)
        # get poses of the sequence images
        poses_im = dh.get_image_poses(world_poses, directions, im_names_all, image_seq, scale)
        # get target images and their poses
        goal_ims = [x.decode("utf-8")+".jpg" for x in targets_data[cat.encode()][scene.encode()]]
        if goal_ims[-1]==".jpg": # last entry might need fixing
            goal_ims = goal_ims[:-1]
        #print("goal_ims:", goal_ims)
        poses_goal = dh.get_image_poses(world_poses, directions, im_names_all, np.asarray(goal_ims), scale)        

        save_dir = save_path+str(i)+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig, ax = plt.subplots(1,2)
        # visualize the point cloud and the target poses
        ax[0].scatter(pcloud[:,0], pcloud[:,2], c=color_cloud/255.0, s=2)
        ax[0].scatter(poses_goal[:,0], poses_goal[:,1], color="green", s=3)
        for j in range(im_list.shape[0]): # add the rest of the steps on the plot
            ax[1].set_title(cat)
            plot_step_nav(ax=ax, k=j, im_list=im_list, poses=poses_im, save_dir=save_dir)
        plt.clf()



def visualize_loc(avd_root, episode_results, scene, save_path):
    _, scale, _, _, _ = dh.load_scene_info(avd_root, scene)
    # Load the pcloud
    scene_path = avd_root + scene + "/" # to read the images
    pcloud, color_cloud = get_pcloud(scene_path=scene_path, scale=scale)
    for i in range(len(episode_results)):
        (image_seq, rel_pose, abs_pose, pred_rel_pose, scene, _) = episode_results[i]
        # in case anyone wants to show the images as well...
        image_seq = np.asarray(image_seq)
        im_list = get_images(imgs_name=image_seq, scene_path=scene_path)

        init_pose = abs_pose[0,:]
        poses_gt = dh.absolute_poses(rel_pose=rel_pose, origin=init_pose)
        poses_pred = dh.absolute_poses(rel_pose=pred_rel_pose, origin=init_pose)

        save_dir = save_path+str(i)+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig, ax = plt.subplots(1,1)
        # visualize the point cloud (with color)
        ax.scatter(pcloud[:,0], pcloud[:,2], c=color_cloud/255.0, s=7)
        # show the init pose
        plot_step_loc(ax=ax, k=0, poses_pred=poses_pred, poses_gt=poses_gt, save_dir=save_dir)
        for j in range(1, im_list.shape[0]): # add the rest of the steps on the plot
            plot_step_loc(ax=ax, k=j, poses_pred=poses_pred, poses_gt=poses_gt, save_dir=save_dir)
        plt.clf()



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)

    episode_results = pickle.load(open(config.episode_results_file, 'rb'))

    if (len(episode_results[0]) > 4 and config.mode=="nav") or (len(episode_results[0]) < 5 and config.mode=="loc"):
        raise Exception("Loaded the wrong episode_results_file for the chosen mode!")

    if config.mode == "nav":
        visualize_nav(config.avd_root, episode_results, config.scene, config.save_path)
    else:
        visualize_loc(config.avd_root, episode_results, config.scene, config.save_path)
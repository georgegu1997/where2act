import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import json
import h5py
import imageio

from copy import deepcopy

import utils
from utils import get_global_position_from_camera
from pointnet2_ops.pointnet2_utils import furthest_point_sample

class ModelWrapper():
    def __init__(self, exp_name, model_epoch, model_version, log_folder = "/home/qiaog/src/where2act/code/logs"):
        # set up device
        device = torch.device('cuda:0')

        # load train config
        train_conf = torch.load(os.path.join(log_folder, exp_name, 'conf.pth'))

        # load model
        model_def = utils.get_model_module(model_version)

        # create models
        network = model_def.Network(train_conf.feat_dim, train_conf.rv_dim, train_conf.rv_cnt)

        # load pretrained model
        print('Loading ckpt from ', os.path.join(log_folder, exp_name, 'ckpts'), model_epoch)
        data_to_restore = torch.load(os.path.join(log_folder, exp_name, 'ckpts', '%d-network.pth' % model_epoch))
        network.load_state_dict(data_to_restore, strict=False)
        print('DONE\n')

        # send to device
        network.to(device)

        # set models to evaluation mode
        network.eval()

        self.device = device
        self.network = network
        self.train_conf = train_conf

    def loadAndPredict(self, folder_path):
        # Load the data
        data = loadData(folder_path)

        gt_movable_link_mask = data['gt_movable_link_mask']
        rgb = data['rgb']

        # sample a pixel to interact
        xs, ys = np.where(gt_movable_link_mask>0)
        if len(xs) == 0:
            print('No Movable Pixel! Quit!')
            exit(1)
        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        marked_rgb = (rgb*255).astype(np.uint8)
        marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)

        # prepare input pc
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = data['id1'], data['id2'], data['pc']
        cam_XYZA = compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, rgb.shape[0], rgb.shape[1])
        pt = cam_XYZA[x, y, :3]
        ptid = np.array([x, y], dtype=np.int32)
        mask = (cam_XYZA[:, :, 3] > 0.5)
        mask[x, y] = False
        pc = cam_XYZA[mask, :3]
        grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
        grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
        pcids = grid_xy[:, mask].T
        pc_movable = (gt_movable_link_mask > 0)[mask]
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        while len(idx) < 30000:
            idx = np.concatenate([idx, idx])
        idx = idx[:30000-1]
        pc = pc[idx, :]
        pc_movable = pc_movable[idx]
        pcids = pcids[idx, :]
        pc = np.vstack([pt, pc])
        pcids = np.vstack([ptid, pcids])
        pc_movable = np.append(True, pc_movable)
        pc[:, 0] -= 5
        pc = torch.from_numpy(pc).unsqueeze(0).to(self.device)

        input_pcid = furthest_point_sample(pc, self.train_conf.num_point_per_shape).long().reshape(-1)
        pc = pc[:, input_pcid, :3]  # 1 x N x 3
        pc_movable = pc_movable[input_pcid.cpu().numpy()]     # N
        pcids = pcids[input_pcid.cpu().numpy()]
        pccolors = rgb[pcids[:, 0], pcids[:, 1]]

        with torch.no_grad():
            # push through unet
            feats = self.network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F

            # First get the actionability score
            pred_ab_score = self.network.inference_action_score(pc)[0] # N
            pred_ab_score = pred_ab_score.cpu().numpy()

            # Then sample the best one, put it on to the first point in pc
            pred_ab_score = pred_ab_score * pc_movable # Ignore non-movable points
            pred_ab_idx = pred_ab_score.argmax() 
            pc[:, 0] = pc[:, pred_ab_idx]

            # Get the action proposal and get the action score
            pred_6d = self.network.inference_actor(pc)[0]  # RV_CNT x 6
            pred_Rs = self.network.actor.bgs(pred_6d.reshape(-1, 3, 2))    # RV_CNT x 3 x 3

            gripper_direction_camera = pred_Rs[0:1, :, 0]
            gripper_forward_direction_camera = pred_Rs[0:1, :, 1]
            
            result_scores = self.network.inference_critic(pc, gripper_direction_camera, gripper_forward_direction_camera, abs_val=True)
            result_scores = result_scores.detach().cpu().numpy()

        best_proposal_idx = result_scores.argmax()

        out = {
            "x": pc[0, 0, 0].item(), 
            "y": pc[0, 0, 1].item(), 
            "action_score": result_scores[best_proposal_idx],
            "actionability_score": pred_ab_score[pred_ab_idx],
            "gripper_direction_camera":  pred_Rs[best_proposal_idx, :, 0].cpu().numpy(),
            "gripper_forward_direction_camera":  pred_Rs[best_proposal_idx, :, 1].cpu().numpy(),
        }

        return out

def loadData(folder_path):
    if folder_path[-1] == "/":
        folder_path = folder_path[:-1]
        
    # Parse the folder_name to get metadata
    folder_name = os.path.basename(folder_path)
    shape_id, category, cnt_id, primact_type, trial_id = folder_name.split("_")
    shape_id = int(shape_id)
    cnt_id = int(cnt_id)
    trial_id = int(trial_id)
    
    # Load the RGB image and the segmentation map
    seg = pickle.load(open(os.path.join(folder_path, "seg.pkl"), "rb"))
    rgb = imageio.imread(os.path.join(folder_path, "rgb.png"))
    gt_movable_link_mask = imageio.imread(os.path.join(folder_path, "interaction_mask.png"))
    # obj_seg = pickle.load(open(os.path.join(folder_path, "obj_seg.pkl"), "rb")) # More finegrained segmentation, not useful

    # The result dictionary 
    result = json.load(open(os.path.join(folder_path, "result.json"), 'r'))

    joints = deepcopy(result['joints'])
    for i, j in enumerate(joints):
        j['pose_cam'] = np.asarray(j['pose_cam'])
        j['pose_global'] = np.asarray(j['pose_global'])
        j['angle'] = result['joint_angles'][i]
        j['angle_lower'] = result['joint_angles_lower'][i]
        j['angle_upper'] = result['joint_angles_upper'][i]

    # point cloud in the camera coordinate system
    xyza_path = os.path.join(folder_path, "cam_XYZA.h5")
    with h5py.File(xyza_path, "r") as f:
        # List all groups
        id1 = np.asarray(f['id1'])
        id2 = np.asarray(f['id2'])
        pc = np.asarray(f['pc'])

    pc_seg = seg[id1, id2]

    out = {
        "shape_id": shape_id, 
        "seg": seg, 
        "rgb": rgb,
        "joints": joints, 
        "pc": pc, 
        "pc_seg": pc_seg,
        "result": result, 
        "id1": id1, 
        "id2": id2, 
        "gt_movable_link_mask": gt_movable_link_mask
    }
    return out

def compute_XYZA_matrix(id1, id2, pts, size1, size2):
    out = np.zeros((size1, size2, 4), dtype=np.float32)
    out[id1, id2, :3] = pts
    out[id1, id2, 3] = 1
    return out

if __name__ == "__main__":
    exp_name = "finalexp-model_all_final-pushing-None-train_all_v1"
    model_epoch = 92
    model_version = "model_3d_legacy"
    folder_path = "../data/gt_data-train_10cats_train_data-pushing/100141_Box_2_pushing_126/"

    wrapper = ModelWrapper(exp_name, model_epoch, model_version)
    out = wrapper.loadAndPredict(folder_path)
    print(out)
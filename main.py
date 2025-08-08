import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from DA2_depth import run_depth_anything_v2
from boostmonodepth_utils import run_boostmonodepth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
# Debug outputs
DEBUG_MODE = os.environ.get('DEBUG_MODE') is not None
DEBUG_DIR = os.environ.get('DEBUG_DIR') or None

# Optional runtime overrides
env_specific = os.environ.get('SPECIFIC_NAME')
if env_specific:
    config['specific'] = env_specific
env_longer_side = os.environ.get('LONGER_SIDE_LEN')
if env_longer_side:
    try:
        config['longer_side_len'] = int(env_longer_side)
    except Exception:
        pass
env_save_ply = os.environ.get('SAVE_PLY')
if env_save_ply is not None:
    config['save_ply'] = str(env_save_ply).lower() in ('1', 'true', 'yes', 'y')
env_encoder = os.environ.get('DEPTH_ANYTHING_ENCODER')
if env_encoder:
    config['depth_anything_encoder'] = env_encoder
env_fast = os.environ.get('FAST_MODE')
if env_fast:
    # fast mode: lower internal sizes/iters but keep fps the same
    config['longer_side_len'] = min(int(config.get('longer_side_len', 768)), 640)
    config['sparse_iter'] = 1
    config['largest_size'] = 320
    config['redundant_number'] = min(int(config.get('redundant_number', 12)), 6)
    config['repeat_inpaint_edge'] = False
    config['extrapolation_thickness'] = 0
    # shrink neighborhoods to reduce graph size
    config['background_thickness'] = max(20, int(config.get('background_thickness', 70) * 0.5))
    config['context_thickness'] = max(40, int(config.get('context_thickness', 140) * 0.5))

# Optional video-related overrides
env_dur = os.environ.get('DURATION_SECONDS')
env_fps = os.environ.get('FPS_OVERRIDE')
env_loop = os.environ.get('LOOP_MODE')
env_speed = os.environ.get('SPEED_MULTIPLIER')
# Enforce constant FPS at 30 regardless of UI
config['fps'] = 30
if env_dur:
    try:
        duration_seconds = max(0.5, float(env_dur))
        config['num_frames'] = int(round(config['fps'] * duration_seconds))
    except Exception:
        pass
if env_loop:
    config['loop_motion'] = True
if env_speed:
    try:
        config['speed_multiplier'] = float(env_speed)
    except Exception:
        pass

# Optional explicit pixel crop after aspect crop
def _read_px(name: str):
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        return max(0, int(float(val)))
    except Exception:
        return None
crop_top_px = _read_px('CROP_TOP_PX')
crop_bottom_px = _read_px('CROP_BOTTOM_PX')
crop_left_px = _read_px('CROP_LEFT_PX')
crop_right_px = _read_px('CROP_RIGHT_PX')
if any(v is not None for v in [crop_top_px, crop_bottom_px, crop_left_px, crop_right_px]):
    config['pixel_crop'] = {
        'top': crop_top_px or 0,
        'bottom': crop_bottom_px or 0,
        'left': crop_left_px or 0,
        'right': crop_right_px or 0,
    }
else:
    # Default crop: 150px on all sides after aspect crop
    config['pixel_crop'] = {'top': 150, 'bottom': 150, 'left': 150, 'right': 150}

# Optional IO path overrides for per-job isolation
env_src = os.environ.get('SRC_FOLDER')
env_depth = os.environ.get('DEPTH_FOLDER')
env_mesh = os.environ.get('MESH_FOLDER')
env_video = os.environ.get('VIDEO_FOLDER')
if env_src:
    config['src_folder'] = env_src
if env_depth:
    config['depth_folder'] = env_depth
if env_mesh:
    config['mesh_folder'] = env_mesh
if env_video:
    config['video_folder'] = env_video

# Progress reporting
PROGRESS_FILE = os.environ.get('PROGRESS_FILE')
def write_progress(percent: int, message: str):
    if not PROGRESS_FILE:
        return
    try:
        import json
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'percent': int(percent), 'message': message}, f)
    except Exception:
        pass
if config['offscreen_rendering'] is True:
    vispy.use(app='egl')
os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['video_folder'], exist_ok=True)
os.makedirs(config['depth_folder'], exist_ok=True)
sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific'])
normal_canvas, all_canvas = None, None

if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
    device = config["gpu_ids"]
else:
    device = "cpu"

print(f"running on device {device}")

for idx in tqdm(range(len(sample_list))):
    write_progress(1, 'Starting')
    depth = None
    sample = sample_list[idx]
    print("Current Source ==> ", sample['src_pair_name'])
    mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
    image = imageio.imread(sample['ref_img_fi'])

    print(f"Running depth extraction at {time.time()}")
    if config.get('use_depth_anything_v2', False):
        run_depth_anything_v2([sample['ref_img_fi']], config['depth_folder'], encoder=config.get('depth_anything_encoder', 'vits'), input_size=518)
    elif config['use_boostmonodepth'] is True:
        run_boostmonodepth(sample['ref_img_fi'], config['src_folder'], config['depth_folder'])
    elif config['require_midas'] is True:
        run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
                  config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=448)
    write_progress(15, 'Depth estimated')

    if 'npy' in config['depth_format']:
        config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
    else:
        config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']
    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        config['gray_image'] = True
    else:
        config['gray_image'] = False
    image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
    depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
    if DEBUG_MODE and DEBUG_DIR:
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            # Save resized input image used downstream
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{sample['tgt_name']}_image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Save raw depth (pre-filter) quick preview
            d0 = (depth - depth.min()) / max(1e-6, (depth.max() - depth.min()))
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{sample['tgt_name']}_depth_pre.png"), (d0*255).astype(np.uint8))
        except Exception:
            pass
    mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
    if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
        write_progress(25, 'Filtering depth')
        vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
        depth = vis_depths[-1]
        if DEBUG_MODE and DEBUG_DIR:
            try:
                # Save depth after filtering—the one actually used by the pipeline
                d1 = (depth - depth.min()) / max(1e-6, (depth.max() - depth.min()))
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{sample['tgt_name']}_depth_post.png"), (d1*255).astype(np.uint8))
            except Exception:
                pass
        model = None
        torch.cuda.empty_cache()
        print("Start Running 3D_Photo ...")
        print(f"Loading edge model at {time.time()}")
        depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                       map_location=torch.device(device))
        depth_edge_model.load_state_dict(depth_edge_weight)
        depth_edge_model = depth_edge_model.to(device)
        depth_edge_model.eval()

        print(f"Loading depth model at {time.time()}")
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                       map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        depth_feat_model = depth_feat_model.to(device)
        print(f"Loading rgb model at {time.time()}")
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                     map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        rgb_model = rgb_model.to(device)
        graph = None


        print(f"Writing depth ply (and basically doing everything) at {time.time()}")
        write_progress(55, 'Building mesh')
        rt_info = write_ply(image,
                              depth,
                              sample['int_mtx'],
                              mesh_fi,
                              config,
                              rgb_model,
                              depth_edge_model,
                              depth_edge_model,
                              depth_feat_model)

        if rt_info is False:
            continue
        rgb_model = None
        color_feat_model = None
        depth_edge_model = None
        depth_feat_model = None
        torch.cuda.empty_cache()
    if config['save_ply'] is True or config['load_ply'] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info


    print(f"Making video at {time.time()}")
    videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
    # Disable extra cropping in final video canvas
    border = None
    write_progress(80, 'Rendering videos')
    normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                        copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['video_folder']),
                        image.copy(), copy.deepcopy(sample['int_mtx']), config, image,
                        videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                        mean_loc_depth=mean_loc_depth)
    write_progress(100, 'Done')

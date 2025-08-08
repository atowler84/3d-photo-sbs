"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
# from monodepth_net import MonoDepthNet
# import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio


def run_depth(img_names, input_path, output_path, model_path, Net, utils, target_w=None):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device (prefer Apple MPS if available)
    device = torch.device("cpu")
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
    except Exception:
        device = torch.device("cpu")
    print("device: %s" % device)

    # load network (fallback to torch.hub MiDaS if local checkpoint missing)
    use_hub = False
    model = None
    try:
        if model_path is None or (isinstance(model_path, str) and not os.path.isfile(model_path)):
            use_hub = True
        if not use_hub:
            model = Net(model_path)
            model.to(device)
            model.eval()
    except Exception:
        use_hub = True
        model = None
    if use_hub:
        print("Falling back to torch.hub MiDaS weights (DPT_Hybrid)")
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        dpt_transform = midas_transforms.dpt_transform

    # get input
    # img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = utils.read_image(img_name)
        w = img.shape[1]
        base_target = 640.
        if target_w is not None:
            try:
                base_target = float(target_w)
            except Exception:
                base_target = 640.
        scale = base_target / max(img.shape[0], img.shape[1])
        target_height, target_width = int(round(img.shape[0] * scale)), int(round(img.shape[1] * scale))
        if use_hub:
            input_batch = dpt_transform((img * 255).astype(np.uint8)).to(device)
            with torch.no_grad():
                pred = midas(input_batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(target_height, target_width), mode="bicubic", align_corners=False
                ).squeeze()
            depth = pred.cpu().numpy().astype(np.float32)
            img = cv2.resize((img * 255).astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            img_input = utils.resize_image(img)
            print(img_input.shape)
            img_input = img_input.to(device)
            # compute
            with torch.no_grad():
                out = model.forward(img_input)
            depth = utils.resize_depth(out, target_width, target_height)
            img = cv2.resize((img * 255).astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_AREA)

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        np.save(filename + '.npy', depth)
        utils.write_depth(filename, depth, bits=2)

    print("finished")


# if __name__ == "__main__":
#     # set paths
#     INPUT_PATH = "image"
#     OUTPUT_PATH = "output"
#     MODEL_PATH = "model.pt"

#     # set torch options
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

#     # compute depth maps
#     run_depth(INPUT_PATH, OUTPUT_PATH, MODEL_PATH, Net, target_w=640)

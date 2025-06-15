import os
import tensorflow as tf
import numpy as np
import cv2
import yaml
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'vo':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

import sys
sys.path.append(project_root)

from model.depth_net import DispNet
from model.pose_net import PoseNet

def visualize_depth(image, depth_map, use_percentile=True):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    depth_map = 1.0 / (depth_map + 1e-6)
    
    if use_percentile:
        vmin = np.percentile(depth_map, 5)
        vmax = np.percentile(depth_map, 95)
        
        depth_map = np.clip(depth_map, vmin, vmax)
        depth_map = (depth_map - vmin) / (vmax - vmin + 1e-6)
    else:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    return np.hstack([image, depth_color])

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1. / scaled_disp
    return depth

def test_depth_model(model_path, test_dir, output_dir, config_path='vo/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        visible_gpus = config.get('Experiment', {}).get('gpus', [0])
        selected_gpus = [gpus[i] for i in visible_gpus if i < len(gpus)]
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        print(f"using GPU: {visible_gpus}")
    
    batch_size = 1
    image_shape = (config['Train']['img_h'], config['Train']['img_w'])
    min_depth = config['Train']['min_depth']
    max_depth = config['Train']['max_depth']
    
    depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='disp_resnet')
    depth_net.build((batch_size, *image_shape, 3))
    
    try:
        depth_net.load_weights(model_path)
        print(f"load model: {model_path}")
    except Exception as e:
        print(f"fail: {e}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = sorted(glob.glob(os.path.join(test_dir, '*.jpg')) + 
                         glob.glob(os.path.join(test_dir, '*.png')))
    


    
    for idx, img_path in enumerate(tqdm(test_images)):
        img = cv2.imread(img_path)

        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        target_size = (config['Train']['img_w'], config['Train']['img_h'])
        if img.shape[:2] != target_size[::-1]:
            img = cv2.resize(img, target_size)
        
        img_norm = img.astype(np.float32) / 255.0
        img_batch = tf.convert_to_tensor(np.expand_dims(img_norm, axis=0))
        
        disp_raw = depth_net(img_batch, training=False)
        disp = disp_raw[0][0].numpy()
        
        disp_min = np.min(disp)
        disp_max = np.max(disp)
        disp_mean = np.mean(disp)
        disp_median = np.median(disp)
        
        print(f"\nimage{idx+1}/{len(test_images)} - {os.path.basename(img_path)}")
        print(f"  Disparity min: {disp_min:.6f}, max: {disp_max:.6f}")
        print(f"  Disparity avg : {disp_mean:.6f}, medi: {disp_median:.6f}")
        
        vis_with_percentile = visualize_depth(img, disp, use_percentile=True)
        vis_full_range = visualize_depth(img, disp, use_percentile=False)
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        vis_percentile_path = os.path.join(output_dir, f"{base_name}_depth_percentile.jpg")
        cv2.imwrite(vis_percentile_path, cv2.cvtColor(vis_with_percentile, cv2.COLOR_RGB2BGR))
        
        vis_full_path = os.path.join(output_dir, f"{base_name}_depth_full.jpg")
        cv2.imwrite(vis_full_path, cv2.cvtColor(vis_full_range, cv2.COLOR_RGB2BGR))
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.hist(disp.flatten(), bins=50)
        plt.title(f'Disparity Histogram - {base_name}')
        plt.xlabel('Disparity')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 1, 2)
        depth = 1.0 / (disp + 1e-6)
        plt.hist(depth.flatten(), bins=50, range=(0, 30))
        plt.title(f'Depth Histogram - {base_name}')
        plt.xlabel('Depth (m)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        hist_path = os.path.join(output_dir, f"{base_name}_histogram.jpg")
        plt.savefig(hist_path)
        plt.close()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Depth Prediction Test')
    parser.add_argument('--model', type=str, required=False,
                        help='Path to the model weights file',
                        default='/workspace/Deep-Visual-SLAM-main/weights/Monodepth2-resnet18-Posenet-mars-depthScale/run_20250517-082354/depth_net_epoch_30.weights.h5')
    parser.add_argument('--test_dir', type=str, required=False, 
                        help='Directory containing test images',
                        default='/workspace/Deep-Visual-SLAM-main/vo/Silsub_0425/camera1/valid/sequence_10')
    parser.add_argument('--output_dir', type=str, default='./test_results', 
                        help='Directory to save output images')
    parser.add_argument('--config', type=str, default='vo/config.yaml', 
                        help='Path to the config file')
    
    args = parser.parse_args()
    
    test_depth_model(args.model, args.test_dir, args.output_dir, args.config)
import os
import cv2
import numpy as np

# Define paths
BASE_DIR = r'D:\EGDnet\dataset'
SCENES = ['Urban', 'Downtown', 'Pillar World']

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_edge_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.magnitude(edge_x, edge_y)
    edge = np.uint8(np.clip(edge, 0, 255))
    return edge

def compute_gradient_map(depth):
    if depth is None:
        raise ValueError("Depth input is None.")
    
    if not isinstance(depth, np.ndarray):
        raise ValueError("Depth input is not a numpy array.")
    
    if depth.size == 0:
        raise ValueError("Depth array is empty.")
    
    # Normalize to 0-255 and convert to uint8
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    if depth is None:
        raise ValueError("cv2.normalize returned None. Check input values.")
    
    depth = depth.astype(np.uint8)

    # Compute gradients using Sobel filter
    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return grad_magnitude

for scene in SCENES:
    image_dir = os.path.join(BASE_DIR, scene, 'Images')
    depth_dir = os.path.join(BASE_DIR, scene, 'Depth')
    edge_dir = os.path.join(BASE_DIR, scene, 'edges')
    grad_dir = os.path.join(BASE_DIR, scene, 'gradients')

    ensure_dir(edge_dir)
    ensure_dir(grad_dir)

    image_files = sorted(os.listdir(image_dir))
    depth_files = sorted(os.listdir(depth_dir))

    for i, (img_file, d_file) in enumerate(zip(image_files, depth_files)):
        img_path = os.path.join(image_dir, img_file)
        depth_path = os.path.join(depth_dir, d_file)

        # Read image and depth
        image = cv2.imread(img_path)
        depth = np.load(depth_path)

        # Generate maps
        edge_map = compute_edge_map(image)
        gradient_map = compute_gradient_map(depth)

        # Save results with 5-digit filenames
        filename = f'{i:05d}.png'
        edge_output = os.path.join(edge_dir, f'edge_{filename}')
        grad_output = os.path.join(grad_dir, f'gradient_{filename}')

        cv2.imwrite(edge_output, edge_map)
        cv2.imwrite(grad_output, gradient_map)

        print(f"[{scene}] Processed {filename}")

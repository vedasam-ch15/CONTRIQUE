import sys
sys.path.append("/work/09519/vedasam_ch/ls6/anaconda3/envs/CONTRIQUE/lib/python3.7/site-packages")

import torch
from torchvision.models import resnet50
from modules.CONTRIQUE_model import CONTRIQUE_model
from torchvision import transforms
import numpy as np
import os
import argparse
import pickle
import csv
from PIL import Image
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # Load scores from CSV file
    scores = load_scores(args.scores_file)

    # Load CONTRIQUE Model
    encoder = resnet50(pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)

    # Initialize an empty list to store features
    features = []

    # Process images and extract features
    for i, (image_file, score) in enumerate(scores.items()):
        image_path = os.path.join(args.image_dir, image_file)

        # Check if image path exists
        if not os.path.exists(image_path):
            print(f"Error: Image path '{image_path}' does not exist.")
            raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

        # Load image
        image = Image.open(image_path)

        # Convert grayscale to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale image by 2
        sz = image.size
        image_2 = image.resize((sz[0] // 2, sz[1] // 2))

        # Load and process depth map
        depth_map_path = os.path.join('/scratch/09519/vedasam_ch/depth_data_testing/SPAQ', image_file)
        depth_map_path = os.path.splitext(depth_map_path)[0] + '.npy'

        # Check if depth map path exists
        if not os.path.exists(depth_map_path):
            print(f"Error: Depth map path '{depth_map_path}' does not exist.")
            raise FileNotFoundError(f"Depth map path '{depth_map_path}' does not exist.")

        depth_map = np.load(depth_map_path)
        depth_map = np.clip(depth_map, 0, None)
        depth_map_resized = cv2.resize(depth_map, (sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)

        depth_map_2_resized = cv2.resize(depth_map, (sz[0] // 2, sz[1] // 2), interpolation=cv2.INTER_CUBIC)

        depth_map_tensor = torch.from_numpy(depth_map_resized).float().unsqueeze(0).unsqueeze(0) / 14000.0
        depth_map_2_tensor = torch.from_numpy(depth_map_2_resized).float().unsqueeze(0).unsqueeze(0) / 14000.0

        # Transform to tensor and append depth map
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
        image_2_tensor = transforms.ToTensor()(image_2).unsqueeze(0).cuda()

        image_tensor = torch.cat([image_tensor, depth_map_tensor.cuda()], dim=1)
        image_2_tensor = torch.cat([image_2_tensor, depth_map_2_tensor.cuda()], dim=1)

        # Extract features
        model.eval()
        with torch.no_grad():
            _, _, _, _, model_feat, model_feat_2, _, _ = model(image_tensor, image_2_tensor)

        feat = np.hstack((model_feat.detach().cpu().numpy(), model_feat_2.detach().cpu().numpy()))
        features.append(feat)

        if (i + 1) % 10 == 0:
            print("Loop index:", i + 1)

    # Save features
    features = np.concatenate(features, axis=0)
    np.save(args.feature_save_path, features)
    print('Done')

def load_scores(scores_file):
    scores = {}
    with open(scores_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_file = row['Image Name']
            scores[image_file] = row
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='MiDaS/SPAQ/TestImage',
                        help='Directory containing images', metavar='')
    parser.add_argument('--model_path', type=str, default='checkpoints_backup_19/checkpoint_depth_19.tar',
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--scores_file', type=str, default='MiDaS/SPAQ/SPAQ_scores.csv',
                        help='Path to CSV file containing scores', metavar='')
    parser.add_argument('--feature_save_path', type=str, default='lin_reg/SPAQ_features_depth.npy',
                        help='Path to save features', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

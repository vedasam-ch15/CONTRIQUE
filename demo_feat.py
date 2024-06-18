import sys
sys.path.append("/work/09519/vedasam_ch/ls6/anaconda3/envs/CONTRIQUE/lib/python3.7/site-packages")

import sys
# print(sys.path)


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

        # Load image
        image = Image.open(image_path)

        # Convert grayscale to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale image by 2
        sz = image.size
        image_2 = image.resize((sz[0] // 2, sz[1] // 2))

        # Transform to tensor
        image = transforms.ToTensor()(image).unsqueeze(0).cuda()
        image_2 = transforms.ToTensor()(image_2).unsqueeze(0).cuda()

        # Extract features
        model.eval()
        with torch.no_grad():
            _, _, _, _, model_feat, model_feat_2, _, _ = model(image, image_2)

        feat = np.hstack((model_feat.detach().cpu().numpy(), model_feat_2.detach().cpu().numpy()))
        features.append(feat)

        if (i + 1) % 10 == 0:
            print("Loop index:", i + 1)

    # Save features
    features = np.concatenate(features, axis=0)
    np.save(args.feature_save_path, features)
    print('Done')

#print("File path:", scores_file)

def load_scores(scores_file):
    scores = {}
    with open(scores_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_file = row['Image name']
            scores[image_file] = row
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='SPAQ/TestImage',
                        help='Directory containing images', metavar='')
    parser.add_argument('--model_path', type=str, default='checkpoints/checkpoint_25.tar',
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--scores_file', type=str, default='SPAQ/SPAQ_scores.csv',
                        help='Path to CSV file containing scores', metavar='')
    parser.add_argument('--feature_save_path', type=str, default='SPAQ/SPAQ_features.npy',
                        help='Path to save features', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

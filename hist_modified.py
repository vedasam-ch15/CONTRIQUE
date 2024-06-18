import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='/path/to/your/numpy/files/',
                        help='Directory containing numpy files', metavar='')
    parser.add_argument('--histogram_save_path', type=str, default='histogram.npy',
                        help='Path to save histogram', metavar='')
    parser.add_argument('--histogram_image_save_path', type=str, default='histogram_plot.png',
                        help='Path to save histogram image', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args

def main(args):
    directory_path = args.image_dir
    numpy_files = [file for file in os.listdir(directory_path) if file.endswith('.npy')]

    batch_size = 100

    min_pixel_value = float('inf')
    max_pixel_value = float('-inf')
    total = 0
    negative = 0
    for file_name in numpy_files:
        file_path = os.path.join(directory_path, file_name)
        depth_map = np.load(file_path)
        min_pixel_value = min(min_pixel_value, depth_map.min())
        max_pixel_value = max(max_pixel_value, depth_map.max())
        total += depth_map.size
        negative += (depth_map < 0).sum()

    num_bins = 15

    device = args.device
    print("Using device:", device)

    hist_values = torch.zeros(num_bins, dtype=torch.float32, device=device)

    for i in range(0, len(numpy_files), batch_size):
        batch_files = numpy_files[i:i + batch_size]
        batch_data = []

        for file_name in batch_files:
            file_path = os.path.join(directory_path, file_name)
            depth_map = np.load(file_path)
            batch_data.append(depth_map.flatten())

        batch_data = np.concatenate(batch_data).flatten()
        batch_data = torch.tensor(batch_data, dtype=torch.float32, device=device)

        batch_hist, bin_edges = np.histogram(batch_data.cpu().numpy(), bins=num_bins,
                                             range=(min_pixel_value, max_pixel_value))

        hist_values += torch.tensor(batch_hist, dtype=torch.float32, device=device)

    hist_np = hist_values.cpu().numpy()

    os.makedirs(os.path.dirname(args.histogram_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.histogram_image_save_path), exist_ok=True)

    np.save(args.histogram_save_path, hist_np)

    max_freq = np.max(hist_np)
    tick_positions_y = np.arange(0, max_freq + 1, step=max_freq // 10)
    tick_labels_y = [f"{int(val / 1e6):,}" for val in tick_positions_y]

    tick_positions_x = np.linspace(min_pixel_value, max_pixel_value, num=num_bins)
    tick_labels_x = [f"{int(val / 1e3):,}" for val in tick_positions_x]
    plt.figure(figsize=(12, 6))
    plt.tight_layout()

    plt.bar(range(num_bins), hist_np, width=1.0, align='edge', alpha=0.7, color='blue')
    plt.xlabel('Pixel Value (in 10^3)', fontsize=8)
    plt.ylabel('Frequency (in 10^6)', fontsize=8)
    plt.title('Histogram : Pixel Values')

    plt.xticks(range(num_bins), tick_labels_x, rotation=90, fontsize=8)

    plt.yticks(tick_positions_y, tick_labels_y, fontsize=8)

    plt.savefig(args.histogram_image_save_path)
    plt.show()

    # Extract the directory name from the path
    directory_name = os.path.basename(os.path.normpath(directory_path))

    # Create a text file name with the directory name
    text_file_name = f'{directory_name}_negative_percentage.txt'

    # Print the percentage of negative values and store the percentage in the text file
    negative_percentage = negative / total
    print("Percentage of negative values: ", negative_percentage)

    with open(text_file_name, 'w') as f:
        f.write(str(negative_percentage))

if __name__ == '__main__':
    args = parse_args()
    main(args)

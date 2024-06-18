import numpy as np
import os
import argparse

def main(image_dir, output_file):
    file_list = os.listdir(image_dir)
    total_pixels = 0
    negative_pixels = 0
    has_nan_values = False

    for file_name in file_list:
        if file_name.endswith('.npy'):  # Check if the file is a numpy file
            file_path = os.path.join(image_dir, file_name)
            data = np.load(file_path)

            total_pixels += data.size

            if np.isnan(data).any():
                has_nan_values = True

            negative_pixels += (data < 0).sum()

    # Calculate the overall percentage of negative pixels
    overall_negative_percentage = (negative_pixels / total_pixels) * 100

    print(f"Overall percentage of negative pixels: {overall_negative_percentage:.2f}%")

    # Store the percentage in the output file
    with open(output_file, 'w') as f:
        f.write(str(overall_negative_percentage))

    if has_nan_values:
        print("NaN values found.")
    else:
        print("You're good! No NaN values found in the whole directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/path/to/your/numpy/files/',
                        help='Directory containing numpy files', metavar='')
    args = parser.parse_args()

    # Construct the output file path based on the image_dir
    output_file = os.path.join(args.image_dir, f'{os.path.basename(args.image_dir)}_negative_percentage.txt')

    main(args.image_dir, output_file)

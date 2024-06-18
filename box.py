import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to calculate the median, Q3, and IQR
def calculate_median_q3_iqr(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return median, q3, iqr

# Argument parser to accept the input directory and output file paths
parser = argparse.ArgumentParser(description="Generate box plot and save median/IQR values for depth maps.")
parser.add_argument("input_directory", help="Path to the directory containing depth maps")
parser.add_argument("box_plot_image_path", help="Path to save the box plot image")
parser.add_argument("text_file_path", help="Path to save the text file with median/Q3/IQR values")

args = parser.parse_args()
base_directory = args.input_directory
box_plot_image_path = args.box_plot_image_path
text_file_path = args.text_file_path

# Extract the directory name
directory_name = os.path.basename(os.path.normpath(base_directory))

# Initialize lists to store median, Q3, and IQR values
all_medians = []
all_q3_values = []
all_iqrs = []

# Set batch size
batch_size = 100

# Loop through the depth maps in the directory in batches
depth_map_values = []
for filename in os.listdir(base_directory):
    if filename.endswith(".npy"):
        depth_map = np.load(os.path.join(base_directory, filename))
        depth_map_values.extend(depth_map.flatten())

        if len(depth_map_values) >= batch_size:
            # Calculate the median, Q3, and IQR for the batch
            median, q3, iqr = calculate_median_q3_iqr(depth_map_values)
            all_medians.append(median)
            all_q3_values.append(q3)
            all_iqrs.append(iqr)
            depth_map_values = []

# Calculate the median, Q3, and IQR for any remaining data
if depth_map_values:
    median, q3, iqr = calculate_median_q3_iqr(depth_map_values)
    all_medians.append(median)
    all_q3_values.append(q3)
    all_iqrs.append(iqr)

# Combine median, Q3, and IQR values from all batches
final_median = np.median(all_medians)
final_q3 = np.median(all_q3_values)
final_iqr = np.median(all_iqrs)

# Calculate Q3 + 1.5 IQR
high_value = final_q3 + 1.5 * final_iqr

# Create the output directories if they don't exist
os.makedirs(os.path.dirname(box_plot_image_path), exist_ok=True)
os.makedirs(os.path.dirname(text_file_path), exist_ok=True)

# Create a box plot
plt.figure(figsize=(6, 6))
plt.boxplot([all_medians], labels=['All Depth Maps'])
plt.ylabel('Median Values')
plt.title('Median Box Plot')

# Save the box plot image
plt.savefig(box_plot_image_path)

# Save final median, Q3, IQR, and high value to a text file
with open(text_file_path, 'w') as file:
    file.write(f"Final Median: {final_median}\n")
    file.write(f"Final Q3: {final_q3}\n")
    file.write(f"Final IQR: {final_iqr}\n")
    file.write(f"High Value (Q3 + 1.5 IQR): {high_value}\n")

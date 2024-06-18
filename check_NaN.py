import numpy as np
import os

def replace_nan_with_average(data):
    height, width = data.shape
    for y in range(height):
        for x in range(width):
            if np.isnan(data[y, x]):
                valid_values = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= y + dy < height and 0 <= x + dx < width and not np.isnan(data[y + dy, x + dx]):
                            valid_values.append(data[y + dy, x + dx])
                if len(valid_values) > 0:
                    data[y, x] = np.mean(valid_values)
    return data

file_list = os.listdir("unlabeled2017/")

max_depth_values = []

has_nan_values = False

for file_name in file_list:
    file_path = os.path.join("unlabeled2017", file_name)
    data = np.load(file_path)
    if np.isnan(data).any():
        has_nan_values = True
        data = replace_nan_with_average(data)

    max_depth_values.append(np.nanmax(data))

global_max_depth = np.nanmax(max_depth_values)

if has_nan_values:
    print("NaN values ound and replaced. Global maximum depth value:", global_max_depth)
else:
    print("You're good! No NaN values found in the whole directory.Global maximum depth value:", global_max_depth)

if not has_nan_values:
    remaining_files = os.listdir("unlabeled2017/")
    for file_name in remaining_files:
        file_path = os.path.join("unlabeled2017", file_name)
        data = np.load(file_path)
        if np.isnan(data).any():
            print("Oops! There are still NaN values in the directory.")
            break
    else:
        print("Great! No more NaN values in the whole directory.")


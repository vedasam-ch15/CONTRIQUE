import pandas as pd
import os

df = pd.read_csv("NaturalBlurSet.csv")

df = df[df['Blur Label'] == 1]

files = df['Image Name'].to_numpy() + ".jpg"

for i in range(files.shape[0]):

    cmd = "scp ./NaturalBlurSet/" + files[i] + " ./blur_image/" + files[i]

    os.system(cmd)
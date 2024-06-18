import os, csv

f=open("filenames.csv",'w')
w=csv.writer(f)
for path, dirs, files in os.walk("./blur_image"):
    for filename in files:
        s = "training_data/UGC_images/blur_image/" + filename
        w.writerow([s])
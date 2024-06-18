import os

ugc_csv_file = 'file_names_ugc_cleaned.csv'

missing_files_ugc = set()

with open(os.path.join('csv_files', ugc_csv_file)) as file:
    lines = file.readlines()
    header = lines[0].strip().split(',')
    file_name_index = header.index('File_names')
    for line in lines[1:]:
        row = line.strip().split(',')
        if len(row) > file_name_index:
            file_name = row[file_name_index]
            if not os.path.exists(file_name):
                missing_files_ugc.add(file_name)

if len(missing_files_ugc) == 0:
    print("All UGC images are found.")
else:
    print("Some UGC images are missing.")

    print("Missing UGC image file names:")
    for file_name in missing_files_ugc:
        print(file_name)
    print(len(missing_files_ugc))

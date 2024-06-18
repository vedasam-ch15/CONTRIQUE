import os

syn_csv_file = 'file_names_syn.csv'

missing_files_syn = set()

with open(os.path.join('csv_files', syn_csv_file)) as file:
    lines = file.readlines()
    header = lines[0].strip().split(',')
    file_name_index = header.index('File_names')
    for line in lines[1:]:
        row = line.strip().split(',')
        if len(row) > file_name_index:
            file_name = row[file_name_index]
            if not os.path.exists(file_name):
                missing_files_syn.add(file_name)

if len(missing_files_syn) == 0:
    print("All synthetic images are found.")
else:
    print("Some synthetic images are missing.")
    print("Missing synthetic image file names:")
    for file_name in missing_files_syn:
        print(file_name)

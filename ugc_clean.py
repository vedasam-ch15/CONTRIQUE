import os
import csv

ugc_csv_file = 'file_names_ugc.csv'
new_csv_file = 'file_names_ugc_cleaned.csv'

missing_files_ugc = set()
rows_to_keep = []

with open(os.path.join('csv_files', ugc_csv_file), 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    file_name_index = header.index('File_names')

    for row in reader:
        if len(row) > file_name_index:
            file_name = row[file_name_index]
            if not os.path.exists(file_name):
                missing_files_ugc.add(file_name)
            else:
                rows_to_keep.append(row)

if len(missing_files_ugc) == 0:
    print("All UGC images are found.")
else:
    print("Some UGC images are missing.")
    print("Missing UGC image file names:")
    for file_name in missing_files_ugc:
        print(file_name)
    print(len(missing_files_ugc))

# Write the cleaned rows to a new CSV file
with open(os.path.join('csv_files', new_csv_file), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    writer.writerows(rows_to_keep)

print(f"Cleaned CSV file '{new_csv_file}' created.")

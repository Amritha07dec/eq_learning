import os

folder_path = '/home/guest/Amritha/3_deg_plots'  # Change this to your folder path

file_count = sum(
    1 for file in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, file))
)

print(f"Number of files in the folder: {file_count}")

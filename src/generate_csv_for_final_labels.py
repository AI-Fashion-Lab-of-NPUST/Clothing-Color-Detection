import os
import csv

# 1. Define the path to your dataset folder
# Replace this with your actual folder path if you are running the script outside the directory
dataset_path = "./Final_Color_Labels"  

# 2. Define the output CSV file name
output_csv = "color_labels.csv"

# 3. Explicitly list your target classes/folders
classes = ["Beige", "Black", "Blue", "Brown", "Green","Grey","Pink","Purple","Red","White","Yellow"]

print("Generating CSV file... Please wait.")

try:
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header row
        writer.writerow(["filename", "label"])
        
        counter = 0
        # Iterate over each category folder
        for label in classes:
            folder_path = os.path.join(dataset_path, label)
            
            # Check if the subfolder actually exists to prevent crashing
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    # Ensure we are mapping files, not subdirectories or system files (like .DS_Store)
                    if os.path.isfile(os.path.join(folder_path, filename)) and not filename.startswith('.'):
                        writer.writerow([filename, label])
                        counter += 1
            else:
                print(f"Warning: Folder for class '{label}' not found at {folder_path}")

    print(f"Success! Created '{output_csv}' with {counter} mapped files.")

except Exception as e:
    print(f"An error occurred: {e}")

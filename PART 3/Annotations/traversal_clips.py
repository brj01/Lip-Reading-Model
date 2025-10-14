import os
import concurrent.futures

def find_files_in_folder(folder_path, file_type):
    """
    Traverse the specified folder and find files of the specified type (mp4 or wav).
    
    :param folder_path: The folder path
    :param file_type: The file type, either 'mp4' or 'wav'
    :return: A list of file paths
    """
    found_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(f".{file_type.lower()}"):
                found_files.append(os.path.join(root, file))
    return found_files

def process_folders(base_folder, file_type, output_txt_file, num_threads=4):
    """
    Use multithreading to traverse folders, find files of the specified type, 
    and write the results to an output text file.
    
    :param base_folder: The root folder path
    :param file_type: The file type, either 'mp4' or 'wav'
    :param output_txt_file: The output text file path
    :param num_threads: The number of threads to use
    """
    # Get all subfolders
    subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    
    all_files = []  # List to store all found file paths
    
    # Use ThreadPoolExecutor to process folders concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit a task for each subfolder to search for files
        futures = [executor.submit(find_files_in_folder, subfolder, file_type) for subfolder in subfolders]
        
        # Collect all file paths
        for future in concurrent.futures.as_completed(futures):
            all_files.extend(future.result())  # Append the result of each task to the list
    
    # Write all file paths to the output text file
    with open(output_txt_file, 'w') as f:
        for file_path in all_files:
            f.write(file_path + '\n')

    print(f"File paths have been saved to {output_txt_file}")

# Example usage
base_folder = "you clips path"  # Replace with your root folder path
file_type = "mp4"  # Or "wav"
output_txt_file = "output_paths.txt"  # Output file path

# Call the function to traverse with multithreading and save the results
process_folders(base_folder, file_type, output_txt_file, num_threads=8)

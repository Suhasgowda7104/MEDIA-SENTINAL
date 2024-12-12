import os
import pandas as pd
from tkinter import Tk, filedialog

def get_all_image_paths(directory):
    # List to store image file paths
    image_paths = []
    
    # Define allowed image file extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    
    # Walking through directory to find image files
    for root, directories, files in os.walk(directory):
        for file in files:
            # Check if the file has an image extension
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    return image_paths

def save_image_paths_to_excel(image_paths, excel_file_name):
    # Ensure the file has the correct extension
    if not excel_file_name.endswith('.xlsx'):
        excel_file_name += '.xlsx'
    
    # Create a DataFrame from the list of image file paths
    df = pd.DataFrame(image_paths, columns=["Image Path"])
    
    # Save DataFrame to an Excel file
    df.to_excel(excel_file_name, index=False)
    print(f"Image paths have been successfully saved to {excel_file_name}")

def select_folder():
    # Open a dialog to select folder
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory()
    return folder_selected

if __name__ == "__main__":
    print("Please select the folder containing the images.")
    
    # Select folder using dialog
    folder_path = select_folder()
    
    # Check if a folder was selected
    if folder_path:
        excel_file_name = input("Enter the Excel file name to save the paths (e.g., image_paths.xlsx): ")
        
        # Get all image paths
        image_paths = get_all_image_paths(folder_path)
        
        # Save the image paths to an Excel file
        if image_paths:
            save_image_paths_to_excel(image_paths, excel_file_name)
        else:
            print("No images found in the selected folder.")
    else:
        print("No folder selected.")

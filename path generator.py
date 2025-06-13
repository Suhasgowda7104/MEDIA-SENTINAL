import os
import pandas as pd
from tkinter import Tk, filedialog

def get_all_media_paths(directory):
    # List to store media file paths
    media_paths = []
    
    # Define allowed image file extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    
    # Define allowed video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    # Combined extensions
    media_extensions = image_extensions + video_extensions
    
    # Walking through directory to find media files
    for root, directories, files in os.walk(directory):
        for file in files:
            # Check if the file has a media extension
            if any(file.lower().endswith(ext) for ext in media_extensions):
                full_path = os.path.join(root, file)
                file_type = 'video' if any(file.lower().endswith(ext) for ext in video_extensions) else 'image'
                media_paths.append((full_path, file_type))
    
    return media_paths

def save_media_paths_to_excel(media_paths, excel_file_name):
    # Ensure the file has the correct extension
    if not excel_file_name.endswith('.xlsx'):
        excel_file_name += '.xlsx'
    
    # Create a DataFrame from the list of media file paths
    df = pd.DataFrame(media_paths, columns=["Media Path", "Media Type"])
    
    # Save DataFrame to an Excel file
    df.to_excel(excel_file_name, index=False)
    print(f"Media paths have been successfully saved to {excel_file_name}")

def select_folder():
    # Open a dialog to select folder
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory()
    return folder_selected

# For backward compatibility
def get_all_image_paths(directory):
    media_paths = get_all_media_paths(directory)
    # Filter to only include images
    image_paths = [path for path, file_type in media_paths if file_type == 'image']
    return image_paths

if __name__ == "__main__":
    print("Please select the folder containing the media files (images and videos).")
    
    # Select folder using dialog
    folder_path = select_folder()
    
    # Check if a folder was selected
    if folder_path:
        excel_file_name = input("Enter the Excel file name to save the paths (e.g., media_paths.xlsx): ")
        
        # Get all media paths
        media_paths = get_all_media_paths(folder_path)
        
        # Save the media paths to an Excel file
        if media_paths:
            save_media_paths_to_excel(media_paths, excel_file_name)
            print(f"Found {len([p for p, t in media_paths if t == 'image'])} images and {len([p for p, t in media_paths if t == 'video'])} videos.")
        else:
            print("No media files found in the selected folder.")
    else:
        print("No folder selected.")

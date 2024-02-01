import os
import cv2
from PIL import Image
import numpy as np
from third_person_man.utils import VideoRecorder
from pathlib import Path

frames_dir = "/media/data/third_person_man/mask_test/JPEGImages/video1"
masks_dir = "/media/data/third_person_man/box_open/test_mask_2"
output_dir = "/media/data/third_person_man/box_open/visualization"

def convert_white_to_transparent(img):
    # Convert PIL Image to NumPy array
    img = img.resize((1280,720), Image.ANTIALIAS)
    img_array = np.array(img)

    # Extract RGB channels
    r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]

    # Create a mask for white pixels
    white_mask = (r == 76) & (g == 76) & (b == 76)

    # Set alpha channel to 0 for white pixels
    a[white_mask] = 0

    # Update alpha channel in the NumPy array
    img_array[:, :, 3] = a

    # Convert back to PIL Image
    img_with_transparency = Image.fromarray(img_array, 'RGBA')

    return img_with_transparency

# # Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get the list of files in the frames directory
frame_files = sorted(os.listdir(frames_dir))

# Loop through each frame
for idx, frame_file in enumerate(frame_files):
    if idx == 0:
        frame_path = os.path.join(output_dir, frame_file.replace(".jpg", "_overlay.png"))
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        overlay_path = os.path.join(output_dir, frame_file.replace(".jpg", "_overlay.png"))
        frame = Image.fromarray(frame)
        frame.save(overlay_path)
        last_frame_file = frame_file
        continue 
    
    else: 
    # Construct the file paths for the frame and mask
        frame_path = os.path.join(output_dir, frame_file.replace(".jpg", "_overlay.png"))
        frame_path = os.path.join(frames_dir, frame_file)
        mask_path = os.path.join(masks_dir, last_frame_file.replace(".jpg", ".png"))
        last_frame_file = frame_file 

    # Load the frame and mask
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    # frame = Image.open(frame_path)
    mask = Image.open(mask_path).convert("RGBA")
    # print(np.array(mask))


    # Convert white background to transparent in the mask
    mask_with_transparency = convert_white_to_transparent(mask)

    # Overlay the mask with transparency on the frame
    overlay = Image.fromarray(frame)
    overlay.paste(mask_with_transparency, (0, 0), mask_with_transparency)

    # Save the overlayed image to the output directory
    overlay_path = os.path.join(output_dir, frame_file.replace(".jpg", "_overlay.png"))
    overlay.save(overlay_path)
    
    




# Create a video from the overlayed images
    
video_recorder = VideoRecorder(
    save_dir = Path("/media/data/third_person_man/mask_test/visualization"),
    fps = 25
    )
frame_files = sorted(os.listdir(output_dir))

for time_step in range(len(frame_files)):
    frame_file = frame_files[time_step]
    frame_path = os.path.join(output_dir, frame_file)
    # Load the frame and mask
    # frame = cv2.imread(frame_path)
    frame = Image.open(frame_path)

    if time_step == 0: 
        video_recorder.init(obs=frame)
    else:
        video_recorder.record_realsense(frame)

    
video_recorder.save_dir= Path(output_dir)
video_name = 'mask_test.mp4'
video_recorder.save(video_name)
import cv2
import numpy as np
from PIL import Image
import os
import zipfile
import io
import mediapipe as mp
import gradio as gr
import tempfile
import shutil

def detect_head(image):
    print("Detecting head in image...")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        print("No face detected.")
        return None
    
    landmarks = results.multi_face_landmarks[0]
    
    h, w, _ = image.shape
    
    forehead_y = int(landmarks.landmark[10].y * h)
    forehead_x = int(landmarks.landmark[10].x * w)
    chin_y = int(landmarks.landmark[152].y * h)
    left_eye_y = int(landmarks.landmark[33].y * h)
    left_ear_x = int(landmarks.landmark[234].x * w)
    right_ear_x = int(landmarks.landmark[454].x * w)
    nose_x = int(landmarks.landmark[1].x * w)
    nose_y = int(landmarks.landmark[1].y * h)

    y_min = nose_y

    found_white_pixel = False
    for y in range(forehead_y, -1, -1):
        pixel_color = image[y, forehead_x]
        
        if all(c >= 250 for c in pixel_color):
            y_min = y  
            found_white_pixel = True
            break

    if not found_white_pixel:
        print("No nearly white pixel found above the nose.")
    
    x_min = max(0, left_ear_x)  
    x_max = min(w, right_ear_x)  

    distance_left_ear_to_nose = abs(left_ear_x - nose_x)
    distance_right_ear_to_nose = abs(right_ear_x - nose_x)

    if distance_left_ear_to_nose < distance_right_ear_to_nose:
        difference = (distance_right_ear_to_nose - distance_left_ear_to_nose) * 0.5  
        difference = int(difference)
        x_max += difference  
        x_max = min(x_max, w)  
    elif distance_left_ear_to_nose > distance_right_ear_to_nose:
        difference = (distance_left_ear_to_nose - distance_right_ear_to_nose) * 0.5  
        difference = int(difference)
        x_min -= difference  
        x_min = max(x_min, 0)  

    chin_y = int(landmarks.landmark[152].y * h)     
    y_max = min(h, chin_y)  

    print("Head detected.")
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def process_image(image, draw_rectangle):
    print("Processing image...")
    
    head = detect_head(image)
    
    if head is None:
        print("No head detected in image. Skipping.")
        return None
    
    x, y, w, h = head
    
    if draw_rectangle:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)

    square_size = max(w, h)
    
    margin = int(square_size * 0.15)
    new_square_size = square_size + 2 * margin
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    left = max(0, center_x - new_square_size // 2)
    top = max(0, center_y - new_square_size // 2)
    
    right = min(image.shape[1], left + new_square_size)
    bottom = min(image.shape[0], top + new_square_size)
    
    final_width = new_square_size
    final_height = int(final_width * 4 / 3)
    
    bottom_padding = final_height - new_square_size
    
    bottom = min(image.shape[0], bottom + bottom_padding)
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    cropped_image = pil_image.crop((left, top, right, bottom))
    
    if cropped_image.width / cropped_image.height != 3/4:
        cropped_image = cropped_image.resize((final_width, final_height), Image.LANCZOS)
    
    print("Image processed.")
    
    return cropped_image

def process_zip(zip_file, draw_rectangle):
    print("Processing ZIP file...")
    
    input_zip = zipfile.ZipFile(zip_file, 'r')
    
    # Create a directory to save processed images
    output_dir_path = os.path.join(os.getcwd(), "processed_images")
    os.makedirs(output_dir_path, exist_ok=True)

    preview_images_paths = []
    
    for filename in input_zip.namelist():
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            with input_zip.open(filename) as file:
                img_data = file.read()
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                processed_img = process_image(img, draw_rectangle)

                if processed_img is not None:
                    output_image_path = os.path.join(output_dir_path, f'processed_{filename}')
                    processed_img.save(output_image_path)  # Save to local folder
                    preview_images_paths.append(output_image_path)

                    print(f"Saved processed image to {output_image_path}")

    print("ZIP file processed.")
    
    return output_dir_path

def cleanup_processed_files(output_dir):
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove individual files
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    # Also remove any existing ZIP files
    zip_file_path = os.path.join(os.getcwd(), "cropped_headshots.zip")
    if os.path.exists(zip_file_path):
        try:
            os.remove(zip_file_path)
        except Exception as e:
            print(f"Error deleting ZIP file {zip_file_path}: {e}")

def gradio_interface(zip_file, draw_rectangle):
    print("Cleaning up old processed files...")
    
    output_dir_path = os.path.join(os.getcwd(), "processed_images")
    cleanup_processed_files(output_dir_path)  # Clean up old files

    print("Received ZIP file. Processing...")
    
    output_dir_path = process_zip(zip_file, draw_rectangle)

    # Create a zip file of processed images for download
    output_zip_path = os.path.join(os.getcwd(), "cropped_headshots.zip")
    with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        for root_dir, _, files in os.walk(output_dir_path):
            for file in files:
                output_zip.write(os.path.join(root_dir, file), arcname=file)

    # Prepare preview image paths for the gallery
    preview_images = [os.path.join(output_dir_path, f) for f in os.listdir(output_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print("Processing complete. Output file ready.")
    return output_zip_path, preview_images

iface = gr.Interface(
   fn=gradio_interface,
   inputs=[
       gr.File(label="Upload ZIP file of photos"),
       gr.Checkbox(label="Check Headshot Region", value=True) 
   ],
   outputs=[
       gr.File(label="Download processed photos", file_count="single", file_types=[".zip"]),
       gr.Gallery(label="Preview of processed photos")
   ],
   title="Headshot Cropping Tool",
   description="Upload a ZIP file containing photos. The tool will crop each photo to focus on the head.",
   allow_flagging="never"
)

iface.launch(share=True)

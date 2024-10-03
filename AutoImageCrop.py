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
    
    # Get specific landmark coordinates for forehead, chin, eyes, ears, and nose
    forehead_y = int(landmarks.landmark[10].y * h)  # Top of forehead
    chin_y = int(landmarks.landmark[152].y * h)     # Bottom of chin
    left_eye_y = int(landmarks.landmark[33].y * h)  # Left eye y-coordinate
    left_ear_x = int(landmarks.landmark[234].x * w)  # Left ear x-coordinate
    right_ear_x = int(landmarks.landmark[454].x * w)  # Right ear x-coordinate
    nose_x = int(landmarks.landmark[1].x * w)         # Nose x-coordinate
    nose_y = int(landmarks.landmark[1].y * h)  # Nose y-coordinate

    # Initialize y_min at the nose position
    y_min = nose_y

    # Check pixels going upward from the nose position
    found_white_pixel = False
    for y in range(nose_y, -1, -1):  # Start checking from the nose position and move upwards
        pixel_color = image[y, nose_x]
        
        if all(c >= 250 for c in pixel_color):  # RGB values close to white
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
    
    if draw_rectangle:  # Check if the rectangle should be drawn
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
    
    output_zip_buffer = io.BytesIO()
    
    output_zip = zipfile.ZipFile(output_zip_buffer, 'w', zipfile.ZIP_DEFLATED)
    
    preview_images = []
    
    for filename in input_zip.namelist():
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            with input_zip.open(filename) as file:
                img_data = file.read()
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                processed_img = process_image(img, draw_rectangle)

                if processed_img is not None:
                    img_byte_arr = io.BytesIO()
                    processed_img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)

                    output_zip.writestr(f'processed_{filename}', img_byte_arr.getvalue())
                    preview_images.append(processed_img)

    output_zip.close()
    
    output_zip_buffer.seek(0)
    
    print("ZIP file processed.")
    
    return output_zip_buffer, preview_images

def gradio_interface(zip_file, draw_rectangle):
    print("Received ZIP file. Processing...")
    
    output_zip, preview_images = process_zip(zip_file, draw_rectangle)

    # Create a temporary file to save the ZIP content.
    temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
   
    with open(temp_file_path, 'wb') as f:
       f.write(output_zip.getvalue())
       
    named_output_path = os.path.join(os.path.dirname(temp_file_path), "cropped_headshots.zip")
    shutil.move(temp_file_path, named_output_path)

    print("Processing complete. Output file ready.")
    return named_output_path, preview_images

iface = gr.Interface(
   fn=gradio_interface,
   inputs=[
       gr.File(label="Upload ZIP file of photos"),
       gr.Checkbox(label="Check Headshot Region", value=True)  # Toggle to draw rectangle
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

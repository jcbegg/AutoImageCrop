import cv2
import numpy as np
from PIL import Image
import os
import zipfile
import io
import gradio as gr

def detect_head(image):
    print("Detecting head in image...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected.")
        return None
    
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = face
    
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(upper_bodies) > 0:
        upper_body = max(upper_bodies, key=lambda b: b[2] * b[3])
        ub_x, ub_y, ub_w, ub_h = upper_body
        y = min(y, ub_y)
        h = max(h, (face[1] + face[3]) - y)
    
    forehead = int(h * 0.2)
    chin = int(h * 0.2)
    
    top = max(0, y - forehead)
    bottom = min(image.shape[0], y + h + chin)
    
    print("Head detected.")
    return (x, top, w, bottom - top)

def process_image(image):
    print("Processing image...")
    height, width = image.shape[:2]
    
    head = detect_head(image)
    
    if head is None:
        print("No head detected in image. Skipping.")
        return None
    
    x, y, w, h = head
    
    square_size = max(w, h)
    
    margin = int(square_size * 0.15)
    new_square_size = square_size + 2 * margin
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    left = max(0, center_x - new_square_size // 2)
    top = max(0, center_y - new_square_size // 2)
    right = min(width, left + new_square_size)
    bottom = min(height, top + new_square_size)
    
    final_width = new_square_size
    final_height = int(final_width * 4 / 3)
    
    bottom_padding = final_height - new_square_size
    
    bottom = min(height, bottom + bottom_padding)
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cropped_image = pil_image.crop((left, top, right, bottom))
    
    if cropped_image.width / cropped_image.height != 3/4:
        cropped_image = cropped_image.resize((final_width, final_height), Image.LANCZOS)
    
    print("Image processed.")
    return cropped_image

def process_zip(zip_file):
    print("Processing ZIP file...")
    input_zip = zipfile.ZipFile(zip_file, 'r')
    output_zip_buffer = io.BytesIO()
    output_zip = zipfile.ZipFile(output_zip_buffer, 'w', zipfile.ZIP_DEFLATED)
    
    preview_images = []
    
    for filename in input_zip.namelist():
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            with input_zip.open(filename) as file:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                processed_img = process_image(img)
                
                if processed_img is not None:
                    img_byte_arr = io.BytesIO()
                    processed_img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    output_zip.writestr(f'processed_{filename}', img_byte_arr)
                    
                    preview_images.append(processed_img)
    
    print("Finalizing ZIP file...")
    output_zip.close()
    output_zip_buffer.seek(0)
    
    print("ZIP file processed.")
    return output_zip_buffer, preview_images

import tempfile
import os
import shutil

# Global variable to store the path of the temporary file
temp_file_path = None

def gradio_interface(zip_file):
    global temp_file_path
    print("Received ZIP file. Processing...")
    output_zip, preview_images = process_zip(zip_file)
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_file_path = temp_file.name
    
    # Write the zip content to the file
    with open(temp_file_path, 'wb') as f:
        f.write(output_zip.getvalue())
    
    # Rename the file to "cropped_headshots.zip"
    named_output_path = os.path.join(os.path.dirname(temp_file_path), "cropped_headshots.zip")
    shutil.move(temp_file_path, named_output_path)
    temp_file_path = named_output_path
    
    print("Processing complete. Output file ready.")
    return named_output_path, preview_images

def cleanup():
    global temp_file_path
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="Upload ZIP file of photos"),
    outputs=[
        gr.File(label="Download processed photos", file_count="single", file_types=[".zip"]),
        gr.Gallery(label="Preview of processed photos")
    ],
    title="Headshot Cropping Tool",
    description="Upload a ZIP file containing photos. The tool will crop each photo to focus on the head, maintaining a 3:4 aspect ratio.",
    allow_flagging="never"
)

iface.launch(share=True)

# Register the cleanup function to be called when the session ends
iface.close = cleanup

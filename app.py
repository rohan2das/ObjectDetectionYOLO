import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
import time
import numpy as np
from datetime import datetime
import torch

def fileuploader(uploaded_file):
    """
    Handle file upload and save to temporary location.
    
    Args:
        uploaded_file: StreamletUploadedFile object
    
    Returns:
        str: Path to the saved file or None if no file uploaded
    """
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file.name
    return None

def filter_results(results, selected_objects, model):
    """
    Filter detection results to only include selected objects.
    
    Args:
        results: YOLO detection results
        selected_objects: List of object names to keep
        model: YOLO model instance containing class names
    
    Returns:
        Modified results with only selected objects
    """
    # If no objects are selected, show all detections
    if not selected_objects:
        return results
        
    # Create a mapping of class names to indices
    name_to_idx = {name: idx for idx, name in model.names.items()}
    
    # Get indices of selected objects
    selected_indices = [name_to_idx[name] for name in selected_objects]
    
    # Filter detections
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Get indices of boxes to keep
            mask = torch.tensor([
                int(box.cls) in selected_indices 
                for box in result.boxes
            ])
            
            # Update boxes with only selected objects
            result.boxes = result.boxes[mask]
            
    return results

def process_frame(frame, model, min_confidence, selected_objects):
    """
    Process a single frame with YOLO model and filter objects.
    
    Args:
        frame: Input frame
        model: YOLO model instance
        min_confidence: Minimum confidence threshold
        selected_objects: List of object names to detect
        
    Returns:
        Annotated frame with filtered detections
    """
    # Run detection
    results = model(frame, conf=min_confidence)
    
    # Filter results to only include selected objects
    filtered_results = filter_results(results, selected_objects, model)
    
    # Plot results
    annotated_frame = filtered_results[0].plot()
    
    return annotated_frame

def process_video(model, video_path, min_confidence, progress_bar, selected_objects):
    """
    Process entire video and save the result.
    
    Args:
        model: YOLO model instance
        video_path (str): Path to input video
        min_confidence (float): Minimum confidence threshold
        progress_bar: Streamlit progress bar object
    
    Returns:
        str: Path to processed video file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = process_frame(frame, model, min_confidence, selected_objects)

            out.write(annotated_frame)
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

    finally:
        cap.release()
        out.release()

    return output_path

def convert_video_to_mp4(input_path):
    """Convert video to MP4 format compatible with web browsers."""
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    os.system(f'ffmpeg -i {input_path} -vcodec libx264 {output_path} -y')
    return output_path

def app():
    # Page configuration
    st.set_page_config(page_title='Object Detection Web App', layout='wide')
    st.title('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome! Choose the detection type below.')

    # Sidebar configuration
    st.sidebar.title("Detection Parameters")
    model_input = st.sidebar.selectbox(
        "Choose model", 
        ["Custom Trained YOLO", "Pretrained YOLO"]
    )

    # Load appropriate model
    model_path = 'human_yolov8x.pt' if model_input == "Custom Trained YOLO" else 'yolov8x.pt'
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Detection parameters
    object_names = list(model.names.values())
    detection_type = st.sidebar.selectbox(
        "Choose Detection Type", 
        ["Image", "Video", "Webcam"]
    )
    selected_objects = st.sidebar.multiselect(
        'Choose objects to detect', 
        object_names,
        'human' if model_input == "Custom Trained YOLO" else 'person'
    )
    min_confidence = st.sidebar.slider('Confidence score', 0.0, 1.0, 0.5)

    # Image Detection
    if detection_type == "Image":
        st.header("Image Detection")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            input_path = fileuploader(uploaded_file)
            
            if st.button('Process Image') and input_path:
                try:
                    # Read image
                    frame = cv2.imread(input_path)
                    # Process with filtered objects
                    annotated_frame = process_frame(frame, model, min_confidence, selected_objects)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.image(annotated_frame_rgb, caption='Processed Image')
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                finally:
                    if os.path.exists(input_path):
                        os.remove(input_path)

    # Video Detection
    elif detection_type == "Video":
        st.header("Video Detection")
        uploaded_file = st.file_uploader("Upload Video", type=['mp4'])
        
        if uploaded_file:
            input_path = fileuploader(uploaded_file)
            
            if st.button('Process Video') and input_path:
                try:
                    progress_text = "Processing video..."
                    progress_bar = st.progress(0, text=progress_text)
                    
                    processed_video_path = process_video(
                        model, 
                        input_path, 
                        min_confidence,
                        progress_bar,
                        selected_objects
                    )
                    
                    final_video_path = convert_video_to_mp4(processed_video_path)
                    
                    progress_bar.progress(1.0, text="Processing complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    
                    with open(final_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    st.download_button(
                        label="Download processed video",
                        data=video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    for path in [input_path, processed_video_path, final_video_path]:
                        if path and os.path.exists(path):
                            os.remove(path)

    # Webcam Detection
    elif detection_type == "Webcam":
        st.header("Webcam Detection")
        st.write("Starting real-time webcam detection. Click 'Stop' to end.")
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("Could not open webcam")
                
            # Create placeholder for webcam feed
            stframe = st.empty()
            
            # Add stop button
            stop_button = st.button("Stop")
            
            try:
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Process frame with YOLO
                    annotated_frame = process_frame(frame, model, min_confidence, selected_objects)
                    
                    # Convert to RGB for display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the processed frame
                    stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Add small delay to prevent high CPU usage
                    time.sleep(0.01)
                    
            finally:
                cap.release()
                stframe.empty()
        except Exception as e:
            st.error(f"Error processing webcam feed: {str(e)}")

if __name__ == "__main__":
    app()
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
import mss
import pygetwindow as gw

# Fix for Streamlit + PyTorch compatibility issue
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

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

def get_available_windows():
    """Get list of available windows for capture."""
    try:
        windows = gw.getAllWindows()
        window_list = []
        for window in windows:
            if window.title and window.title.strip() and window.visible:
                window_info = {
                    'title': window.title,
                    'left': window.left,
                    'top': window.top,
                    'width': window.width,
                    'height': window.height
                }
                window_list.append(window_info)
        return window_list
    except Exception as e:
        st.error(f"Error getting windows: {e}")
        return []

def get_available_monitors():
    """Get list of available monitors."""
    try:
        with mss.mss() as sct:
            monitors = sct.monitors[1:]  # Skip the "All in One" monitor
            return monitors
    except Exception as e:
        st.error(f"Error getting monitors: {e}")
        return []

def capture_screen_area(monitor_info=None, window_info=None):
    """
    Capture screen area (monitor or window).
    
    Args:
        monitor_info: Monitor information dict
        window_info: Window information dict
    
    Returns:
        numpy array: Captured image as BGR array
    """
    try:
        with mss.mss() as sct:
            if window_info:
                # Capture specific window
                w = gw.getWindowsWithTitle(window_info['title'])[0]
                w.activate()
                capture_area = {
                    "top": window_info['top'],
                    "left": window_info['left'],
                    "width": window_info['width'],
                    "height": window_info['height']
                }
            elif monitor_info:
                # Capture specific monitor
                capture_area = monitor_info
            else:
                # Capture primary monitor
                capture_area = sct.monitors[1]
            
            # Capture the screen
            screenshot = sct.grab(capture_area)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Convert to numpy array (BGR format for OpenCV)
            img_array = np.array(img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
    except Exception as e:
        st.error(f"Error capturing screen: {e}")
        return None

def screen_capture_detection():
    """Screen capture detection interface."""
    st.header("Screen Capture Detection")
    
    # Capture type selection
    capture_type = st.selectbox(
        "Choose capture type:",
        ["Select Monitor", "Select Window", "Custom Area"]
    )
    
    if capture_type == "Select Monitor":
        monitors = get_available_monitors()
        if monitors:
            monitor_options = []
            for i, monitor in enumerate(monitors):
                monitor_options.append(f"Monitor {i+1} ({monitor['width']}x{monitor['height']})")
            
            selected_monitor_idx = st.selectbox("Choose monitor:", range(len(monitor_options)), 
                                              format_func=lambda x: monitor_options[x])
            selected_monitor = monitors[selected_monitor_idx]
        else:
            st.error("No monitors found")
            return None, None
            
        return selected_monitor, None
    
    elif capture_type == "Select Window":
        if st.button("Refresh Window List"):
            st.session_state.windows_refreshed = True
            
        windows = get_available_windows()
        if windows:
            window_titles = [f"{w['title'][:50]}..." if len(w['title']) > 50 else w['title'] 
                           for w in windows]
            
            selected_window_idx = st.selectbox("Choose window:", range(len(window_titles)),
                                             format_func=lambda x: window_titles[x])
            selected_window = windows[selected_window_idx]
            
            # Show window info
            st.info(f"Selected: {selected_window['title']}")
            st.write(f"Position: ({selected_window['left']}, {selected_window['top']})")
            st.write(f"Size: {selected_window['width']} x {selected_window['height']}")
            
        else:
            st.error("No windows found")
            return None, None
            
        return None, selected_window
    
    else:  # Custom Area
        st.write("Define custom capture area:")
        col1, col2 = st.columns(2)
        with col1:
            left = st.number_input("Left", value=0, min_value=0)
            top = st.number_input("Top", value=0, min_value=0)
        with col2:
            width = st.number_input("Width", value=800, min_value=1)
            height = st.number_input("Height", value=600, min_value=1)
        
        custom_area = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }
        return custom_area, None

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
    model_path = 'human_yolov8n.pt' if model_input == "Custom Trained YOLO" else 'yolov8n.pt'
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Detection parameters
    object_names = list(model.names.values())
    detection_type = st.sidebar.selectbox(
        "Choose Detection Type", 
        ["Image", "Video", "Camera", "Screen Capture"]
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
                    # Save processed image for download
                    processed_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                    cv2.imwrite(processed_image_path, annotated_frame)
                    # Image to be included in the download
                    with open(processed_image_path, 'rb') as img_file:
                        img_bytes = img_file.read()
                    # Get original filename without extension
                    original_name = Path(uploaded_file.name).stem
                    download_filename = f"{original_name}_processed.jpg"

                    st.download_button(
                        label="Download processed Image",
                        data=img_bytes,
                        file_name=download_filename,
                        mime="image/jpeg"
                    )

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
            cap = cv2.VideoCapture(1)
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
                    stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Add small delay to prevent high CPU usage
                    time.sleep(0.01)
                    
            finally:
                cap.release()
                stframe.empty()
        except Exception as e:
            st.error(f"Error processing webcam feed: {str(e)}")
    
    elif detection_type == "Screen Capture":
        monitor_info, window_info = screen_capture_detection()
        
        if monitor_info or window_info:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Capture Once"):
                    try:
                        # Capture single frame
                        frame = capture_screen_area(monitor_info, window_info)
                        if frame is not None:
                            # Process with YOLO
                            annotated_frame = process_frame(frame, model, min_confidence, selected_objects)
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            st.image(annotated_frame_rgb, caption='Screen Capture Detection')
                    except Exception as e:
                        st.error(f"Error in screen capture: {e}")
            
            with col2:
                if st.button("Start Real-time Capture"):
                    st.session_state.screen_capture_running = True
                if st.button("Stop Real-time Capture"):
                    st.session_state.screen_capture_running = False
            
            # Real-time screen capture
            if st.session_state.get('screen_capture_running', False):
                stframe = st.empty()
                fps_display = st.empty()
                
                try:
                    fps_counter = 0
                    start_time = time.time()
                    
                    while st.session_state.get('screen_capture_running', False):
                        # Capture screen
                        frame = capture_screen_area(monitor_info, window_info)
                        
                        if frame is not None:
                            # Process with YOLO
                            annotated_frame = process_frame(frame, model, min_confidence, selected_objects)
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display frame
                            stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                            
                            # Calculate and display FPS
                            fps_counter += 1
                            if fps_counter % 10 == 0:  # Update FPS every 10 frames
                                elapsed = time.time() - start_time
                                fps = fps_counter / elapsed
                                fps_display.write(f"FPS: {fps:.1f}")
                        
                        # Small delay to prevent excessive CPU usage
                        time.sleep(0.1)  # Adjust for desired frame rate
                        
                except Exception as e:
                    st.error(f"Error in real-time screen capture: {e}")
                finally:
                    if 'stframe' in locals():
                        stframe.empty()
                    if 'fps_display' in locals():
                        fps_display.empty()

if __name__ == "__main__":
    app()
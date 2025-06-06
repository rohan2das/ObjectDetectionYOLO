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
import pygetwindow as gw

# Windows-specific imports for direct window capture
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    from ctypes import windll
    import ctypes
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False
    print("Windows API not available. Some features will be limited.")

# Alternative: Using MSS for faster capture
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("MSS not available. Install with: pip install mss")

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

def capture_screen_area_mss(monitor_info=None, window_info=None):
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

def capture_screen_area_win32(monitor_info=None, window_info=None):
    """
    Method 3: Windows API Direct Window Capture
    - Can capture hidden/occluded windows on Windows
    - Captures actual window content, not screen region
    """
    if not WINDOWS_AVAILABLE:
        return None
        
    try:
        # Try to find window by title (exact match first)
        hwnd = win32gui.FindWindow(None, window_info['title'])
        
        # If exact match fails, search through all windows
        if not hwnd:
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_info['title'] in window_title or window_title in window_info['title']:
                        windows.append(hwnd)
                return True
            
            windows_list = []
            win32gui.EnumWindows(enum_windows_callback, windows_list)
            if windows_list:
                hwnd = windows_list[0]
            else:
                print(f"Could not find window: {window_info['title']}")
                return None
        
        # Get window dimensions
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        
        if width <= 0 or height <= 0:
            print(f"Invalid window dimensions: {width}x{height}")
            return None
        
        # Method 3A: Using ctypes and PrintWindow (best for hidden windows)
        try:
            # Get window device context
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Use ctypes to call PrintWindow
            result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
            
            if result == 0:
                # PrintWindow failed, try BitBlt as fallback
                print("PrintWindow failed, trying BitBlt...")
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            signed_ints_array = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(signed_ints_array, dtype='uint8')
            img.shape = (height, width, 4)
            
            # Clean up resources
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            
            # Convert BGRA to BGR and handle potential issues
            if img.size > 0:
                bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return bgr_img
            else:
                print("Empty image captured")
                return None
                
        except Exception as inner_e:
            print(f"PrintWindow method failed: {inner_e}")
            # Try alternative method
            return capture_method_3b_alternative(hwnd, width, height)
            
    except Exception as e:
        print(f"Windows API capture error: {e}")
        return None
    
def capture_method_3b_alternative(hwnd, width, height):
    """Alternative Windows API method using different approach"""
    try:
        # Get device context differently
        hdc = windll.user32.GetDC(hwnd)
        if not hdc:
            return None
            
        # Create compatible DC and bitmap
        mdc = windll.gdi32.CreateCompatibleDC(hdc)
        hbitmap = windll.gdi32.CreateCompatibleBitmap(hdc, width, height)
        windll.gdi32.SelectObject(mdc, hbitmap)
        
        # Try different PrintWindow flags
        # PW_CLIENTONLY = 1, PW_RENDERFULLCONTENT = 2, PW_CLIENTONLY | PW_RENDERFULLCONTENT = 3
        success = windll.user32.PrintWindow(hwnd, mdc, 0)
        
        if not success:
            # Fallback to BitBlt
            windll.gdi32.BitBlt(mdc, 0, 0, width, height, hdc, 0, 0, win32con.SRCCOPY)
        
        # Get bitmap data
        bmp_info = ctypes.create_string_buffer(40)  # BITMAPINFOHEADER size
        ctypes.memset(bmp_info, 0, 40)
        ctypes.c_uint32.from_buffer(bmp_info, 0).value = 40  # biSize
        ctypes.c_int32.from_buffer(bmp_info, 4).value = width  # biWidth
        ctypes.c_int32.from_buffer(bmp_info, 8).value = -height  # biHeight (negative for top-down)
        ctypes.c_uint16.from_buffer(bmp_info, 12).value = 1  # biPlanes
        ctypes.c_uint16.from_buffer(bmp_info, 14).value = 32  # biBitCount
        
        # Create buffer for image data
        buffer_size = width * height * 4
        image_buffer = ctypes.create_string_buffer(buffer_size)
        
        # Get the bitmap bits
        windll.gdi32.GetDIBits(hdc, hbitmap, 0, height, image_buffer, bmp_info, 0)
        
        # Clean up
        windll.gdi32.DeleteObject(hbitmap)
        windll.gdi32.DeleteDC(mdc)
        windll.user32.ReleaseDC(hwnd, hdc)
        
        # Convert to numpy array
        img = np.frombuffer(image_buffer.raw, dtype='uint8')
        img = img.reshape((height, width, 4))
        
        # Convert BGRA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    except Exception as e:
        print(f"Alternative Windows capture failed: {e}")
        return None

def screen_capture_detection():
    """Screen capture detection interface."""
    st.header("Screen Capture Detection")
    
    # Capture type selection
    capture_type = st.selectbox(
        "Choose capture type:",
        ["Select Window", "Select Monitor"]
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
        
        capture_method=st.selectbox(
                "Choose capture method:",
                ["win32api", "mss"]
            )

        if monitor_info or window_info:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Capture Once"):
                    try:
                        # Capture single frame
                        frame = capture_screen_area_win32(monitor_info, window_info) if capture_method == "win32api" else capture_screen_area_mss(monitor_info, window_info)
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
                        frame = capture_screen_area_win32(monitor_info, window_info) if capture_method == "win32api" else capture_screen_area_mss(monitor_info, window_info)
                        
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
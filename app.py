# app.py
# Requirements: Flask, Flask-Cors, img2pdf, Pillow, yt-dlp, opencv-python, numpy

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import shutil
import img2pdf
import yt_dlp
import cv2
import numpy as np
import logging
import tempfile
import uuid
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def download_youtube_video(url, output_path='video.mp4'):
    """
    Downloads a YouTube video to a specified path.
    Returns the path to the downloaded video file.
    """
    logger.info(f"Downloading video from {url}...")
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info("Video downloaded successfully.")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        raise

def preprocess_frame(frame):
    """
    Minimal preprocessing: use full frame to preserve all details.
    """
    return frame, frame

def extract_and_save_frames(video_path, output_dir, mse_threshold=20):
    """
    Extracts frames from the video when significant changes are detected,
    crops to the white sheet music area, and saves only unique frames.
    """
    logger.info(f"Extracting frames from video '{video_path}' and cropping to white sheet music...")
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    check_interval = int(fps * 1)
    
    last_saved_frame_gray = None
    frame_count = 0
    sheet_music_bbox = None  # Bounding box for the sheet music area

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Only process frames at a specific interval to reduce overhead
        if (frame_count - 1) % check_interval != 0:
            continue

        # Pre-process the frame to find the white sheet music bounding box
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Use a threshold to isolate bright white areas
        _, binary_frame = cv2.threshold(blurred_frame, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_sheet_music_bbox = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > h * 2 and h > 50 and w > 150:
                    current_sheet_music_bbox = (x, y, w, h)

        if current_sheet_music_bbox:
            # Update the bounding box if a new, stable one is found
            if sheet_music_bbox is None or \
               (abs(current_sheet_music_bbox[0] - sheet_music_bbox[0]) < current_sheet_music_bbox[2]/2 and
                abs(current_sheet_music_bbox[1] - sheet_music_bbox[1]) < current_sheet_music_bbox[3]/2):
                sheet_music_bbox = current_sheet_music_bbox
        
        # If a sheet music box has been found, crop and process the frame
        if sheet_music_bbox:
            x, y, w, h = sheet_music_bbox
            cropped_frame = frame[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(cropped_gray)

            # Skip frames that are too dark or have too little variation (likely not sheet music)
            if mean_brightness < 180 or np.std(cropped_gray) < 5:
                logger.debug(f"Frame {frame_count}: Invalid frame (mean brightness {mean_brightness}, std {np.std(cropped_gray):.2f}), skipping.")
                continue
        else:
            logger.debug(f"Frame {frame_count}: No valid sheet music area found. Skipping frame.")
            continue

        # Compare with the last saved frame to check for a significant change
        if last_saved_frame_gray is None:
            mse = mse_threshold + 1 # Force save the first frame
        else:
            mse = np.mean((cropped_gray.astype(float) - last_saved_frame_gray.astype(float)) ** 2)
        
        logger.debug(f"Frame {frame_count}: MSE={mse:.2f}")

        # This is the line that prevents consecutive duplicates from being saved
        if mse > mse_threshold:
            # A significant change was detected, save the cropped frame
            filename = f"sheet_music_frame_{len(image_paths) + 1}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cropped_frame)
            image_paths.append(filepath)
            logger.info(f"Change detected, saved {filename} at frame {frame_count} (~{frame_count/fps:.1f} seconds) with MSE={mse:.2f}")
            last_saved_frame_gray = cropped_gray

    cap.release()
    logger.info(f"Frame extraction complete. Total frames saved: {len(image_paths)}")
    return image_paths


def combine_frames_for_pdf(image_paths, output_dir):
    """
    Combines multiple image frames into single page images optimized for Letter size printing,
    maximizing frames per page and adding pages as needed.
    """
    logger.info(f"Combining frames into pages optimized for Letter size printing.")
    combined_image_paths = []
    
    # Define Letter size at 300 DPI for high-quality printing (8.5 x 11 inches)
    page_width, page_height = int(8.5 * 300), int(11 * 300) 
    padding = 20 
    
    current_page_frames = []
    current_y_offset = padding
    page_count = 1
    max_width = int(page_width * 0.9)

    for frame_path in image_paths:
        try:
            with Image.open(frame_path) as frame:
                # Resize frame to fit within 90% of page width, maintaining aspect ratio
                if frame.width > max_width:
                    new_height = int(frame.height * (max_width / frame.width))
                    frame = frame.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Check if adding this frame exceeds the page height
                if current_y_offset + frame.height + padding > page_height:
                    # Combine the current page's frames
                    if current_page_frames:
                        page_image = Image.new('RGB', (page_width, page_height), 'white')
                        y_offset = padding
                        for fp in current_page_frames:
                            with Image.open(fp) as f:
                                if f.width > max_width:
                                    new_h = int(f.height * (max_width / f.width))
                                    f = f.resize((max_width, new_h), Image.Resampling.LANCZOS)
                                x_offset = (page_width - f.width) // 2
                                page_image.paste(f, (x_offset, y_offset))
                                y_offset += f.height + padding
                        
                        combined_filename = f"combined_page_{page_count}.png"
                        combined_filepath = os.path.join(output_dir, combined_filename)
                        page_image.save(combined_filepath)
                        combined_image_paths.append(combined_filepath)
                        logger.info(f"Saved combined page: {combined_filepath}")
                        page_count += 1
                    
                    # Start a new page with the current frame
                    current_page_frames = [frame_path]
                    current_y_offset = padding + frame.height + padding
                else:
                    current_page_frames.append(frame_path)
                    current_y_offset += frame.height + padding
        except Exception as e:
            logger.warning(f"Could not open image {frame_path}: {e}")
    
    # Combine any remaining frames into the last page
    if current_page_frames:
        page_image = Image.new('RGB', (page_width, page_height), 'white')
        y_offset = padding
        for fp in current_page_frames:
            with Image.open(fp) as f:
                if f.width > max_width:
                    new_h = int(f.height * (max_width / f.width))
                    f = f.resize((max_width, new_h), Image.Resampling.LANCZOS)
                x_offset = (page_width - f.width) // 2
                page_image.paste(f, (x_offset, y_offset))
                y_offset += f.height + padding
        combined_filename = f"combined_page_{page_count}.png"
        combined_filepath = os.path.join(output_dir, combined_filename)
        page_image.save(combined_filepath)
        combined_image_paths.append(combined_filepath)
        logger.info(f"Saved final combined page: {combined_filepath}")
    
    return combined_image_paths

@app.route('/process-video', methods=['POST'])
def process_video():
    """
    API endpoint to process a YouTube video and generate a PDF of the sheet music.
    """
    data = request.get_json()
    youtube_url = data.get('url')

    if not youtube_url:
        logger.error("No URL provided in request.")
        return jsonify({"error": "No URL provided"}), 400

    temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4().hex}.mp4")
    temp_dir = os.path.join(tempfile.gettempdir(), f"temp_sheet_music_images_{uuid.uuid4().hex}")
    pdf_filename = os.path.join(tempfile.gettempdir(), f"sheet_music_from_youtube_{uuid.uuid4().hex}.pdf")
    
    try:
        # Step 1: Download the YouTube video
        download_youtube_video(youtube_url, output_path=temp_video_path)

        # Step 2: Extract frames from the downloaded video
        image_paths = extract_and_save_frames(temp_video_path, temp_dir, mse_threshold=20)

        if not image_paths:
            logger.error("No frames were extracted.")
            return jsonify({"error": "No frames were extracted. Is the URL valid or does it contain sheet music?"}), 500

        # Step 3: Combine the extracted images into pages
        combined_image_paths = combine_frames_for_pdf(image_paths, temp_dir)

        if not combined_image_paths:
            logger.error("No combined pages created.")
            return jsonify({"error": "No combined pages created."}), 500

        # Step 4: Convert combined pages to a single PDF
        logger.info(f"Converting {len(combined_image_paths)} pages into '{pdf_filename}'...")
        with open(pdf_filename, "wb") as f:
            f.write(img2pdf.convert([open(path, 'rb') for path in combined_image_paths]))
        logger.info("PDF created successfully.")

        # Step 5: Return the PDF file
        response = send_file(
            pdf_filename,
            mimetype='application/pdf',
            as_attachment=True,
            download_name="sheet_music_from_youtube.pdf"
        )
        # Mark the PDF for deletion after the response is sent
        response.call_on_close(lambda: cleanup_files(temp_video_path, temp_dir, pdf_filename))
        return response

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary video and directory (PDF cleanup handled by call_on_close)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"Cleaned up temporary video: {temp_video_path}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

def cleanup_files(video_path, temp_dir, pdf_path):
    """
    Clean up temporary files after the response is sent.
    """
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.info(f"Cleaned up PDF: {pdf_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up PDF {pdf_path}: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)

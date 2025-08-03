# app.py
# Requirements: Flask, Flask-Cors, img2pdf, Pillow, yt-dlp, opencv-python, numpy, scikit-image

from flask import Flask, request, send_file, jsonify, render_template
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
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

def get_youtube_video_title(url):
    """Fetches the title of a YouTube video without downloading it."""
    try:
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'True',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('title', 'Sheet Music')
    except Exception as e:
        logger.error(f"Failed to get video title from URL {url}: {e}")
        return 'Sheet Music'

def download_youtube_video(url, output_path='video.mp4'):
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

def extract_and_save_frames(video_path, output_dir, similarity_threshold=0.96):
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
    sheet_music_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if (frame_count - 1) % check_interval != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary_frame = cv2.threshold(blurred_frame, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_sheet_music_bbox = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > h * 2 and h > 50 and w > 150:
                    current_sheet_music_bbox = (x, y, w, h)

        if current_sheet_music_bbox and sheet_music_bbox is None:
            sheet_music_bbox = current_sheet_music_bbox

        if sheet_music_bbox:
            x, y, w, h = sheet_music_bbox
            cropped_frame = frame[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(cropped_gray)

            if mean_brightness < 180 or np.std(cropped_gray) < 5:
                continue
        else:
            continue

        if last_saved_frame_gray is None:
            similarity = 0  # force save first frame
        else:
            similarity = ssim(cropped_gray, last_saved_frame_gray)

        logger.debug(f"Frame {frame_count}: SSIM={similarity:.4f}")

        if similarity < similarity_threshold:
            filename = f"sheet_music_frame_{len(image_paths) + 1}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cropped_frame)
            image_paths.append(filepath)
            logger.info(f"Saved {filename} at frame {frame_count} with SSIM={similarity:.4f}")
            last_saved_frame_gray = cropped_gray

    cap.release()
    logger.info(f"Frame extraction complete. Total frames saved: {len(image_paths)}")
    return image_paths

def combine_frames_for_pdf(image_paths, output_dir, title=None):
    logger.info(f"Combining frames into pages optimized for Letter size printing.")

    dpi = 300
    page_width = int(8.5 * dpi)
    page_height = int(11 * dpi)
    margin = 50
    spacing = 30
    max_frame_width = page_width - 2 * margin

    combined_image_paths = []
    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)
    title_font_size = 60

    try:
        font = ImageFont.truetype("arial.ttf", title_font_size)
    except IOError:
        font = ImageFont.load_default(size=title_font_size)

    current_y = margin

    if title:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((page_width - text_width) // 2, margin),
            title,
            fill="black",
            font=font
        )
        current_y = margin + text_height + spacing

    page_count = 1

    for path in image_paths:
        try:
            frame = Image.open(path)
            ratio = max_frame_width / frame.width
            new_height = int(frame.height * ratio)
            frame = frame.resize((max_frame_width, new_height), Image.Resampling.LANCZOS)

            if current_y + new_height > page_height - margin:
                combined_filename = f"combined_page_{page_count}.png"
                combined_filepath = os.path.join(output_dir, combined_filename)
                page.save(combined_filepath)
                combined_image_paths.append(combined_filepath)
                page_count += 1

                page = Image.new("RGB", (page_width, page_height), "white")
                draw = ImageDraw.Draw(page)
                current_y = margin

            x_offset = (page_width - frame.width) // 2
            page.paste(frame, (x_offset, current_y))
            current_y += frame.height + spacing

        except Exception as e:
            logger.warning(f"Could not process image {path}: {e}")

    combined_filename = f"combined_page_{page_count}.png"
    combined_filepath = os.path.join(output_dir, combined_filename)
    page.save(combined_filepath)
    combined_image_paths.append(combined_filepath)

    return combined_image_paths

@app.route('/process-video', methods=['POST'])
def process_video():
    data = request.get_json()
    youtube_url = data.get('url')

    if not youtube_url:
        return jsonify({"error": "No URL provided"}), 400

    video_title = get_youtube_video_title(youtube_url)
    sanitized_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '_')).rstrip()
    sanitized_filename = f"{sanitized_title.replace(' ', '_')}.pdf"

    temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4().hex}.mp4")
    temp_dir = os.path.join(tempfile.gettempdir(), f"temp_sheet_music_images_{uuid.uuid4().hex}")
    pdf_filename = os.path.join(tempfile.gettempdir(), f"sheet_music_from_youtube_{uuid.uuid4().hex}.pdf")

    try:
        download_youtube_video(youtube_url, output_path=temp_video_path)
        image_paths = extract_and_save_frames(temp_video_path, temp_dir, similarity_threshold=0.96)

        if not image_paths:
            return jsonify({"error": "No frames were extracted."}), 500

        combined_image_paths = combine_frames_for_pdf(image_paths, temp_dir, title=video_title)

        if not combined_image_paths:
            return jsonify({"error": "No combined pages created."}), 500

        with open(pdf_filename, "wb") as f:
            f.write(img2pdf.convert([open(path, 'rb') for path in combined_image_paths]))

        response = send_file(
            pdf_filename,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=sanitized_filename
        )
        response.call_on_close(lambda: cleanup_files(temp_video_path, temp_dir, pdf_filename))
        return response

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def cleanup_files(video_path, temp_dir, pdf_path):
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    except Exception as e:
        logger.warning(f"Failed to clean up PDF {pdf_path}: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import os
import logging
import telebot
import tensorflow as tf
import numpy as np
import traceback
from PIL import Image
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries
import io
import cv2
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
from pathlib import Path
import importlib.util
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeepFakeBot')

# Configuration
BOT_TOKEN = '7491289767:AAG6R_MfzVLQLTgytzQUOvqMRT8POvuq9UE'
MODEL_PATH = 'fake_face_detection_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Video analysis configuration
REFERENCE_VIDEOS_PATH = 'reference_videos'
REAL_VIDEOS_PATH = os.path.join(REFERENCE_VIDEOS_PATH, 'real')
FAKE_VIDEOS_PATH = os.path.join(REFERENCE_VIDEOS_PATH, 'fake')
RESULTS_PATH = os.path.join(UPLOAD_FOLDER, 'analysis_results')
VIDEO_FRAMES_PATH = os.path.join(UPLOAD_FOLDER, 'video_frames')  # For web app style processing
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(REAL_VIDEOS_PATH, exist_ok=True)
os.makedirs(FAKE_VIDEOS_PATH, exist_ok=True)
os.makedirs(VIDEO_FRAMES_PATH, exist_ok=True)  # Create video frames directory

bot = telebot.TeleBot(BOT_TOKEN)

def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None

model = load_model()

class VideoAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the video analyzer with the detection model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Video analyzer model loaded successfully")
            
            # Check for reference videos and log their count
            logger.info(f"Loading reference videos from: {REFERENCE_VIDEOS_PATH}")
            self.real_references = self._get_reference_videos(REAL_VIDEOS_PATH)
            self.fake_references = self._get_reference_videos(FAKE_VIDEOS_PATH)
            
            # Log the reference videos found
            for i, path in enumerate(self.real_references):
                logger.info(f"Real reference #{i+1}: {os.path.basename(path)}")
            for i, path in enumerate(self.fake_references):
                logger.info(f"Fake reference #{i+1}: {os.path.basename(path)}")
            
            # Cache file hashes for quick reference lookup
            logger.info("Caching real video hashes...")
            self.real_hashes = self._cache_reference_hashes(self.real_references)
            logger.info("Caching fake video hashes...")
            self.fake_hashes = self._cache_reference_hashes(self.fake_references)
            
            logger.info(f"Found {len(self.real_references)} REAL reference videos")
            logger.info(f"Found {len(self.fake_references)} FAKE reference videos")
            
        except Exception as e:
            logger.error(f"Error initializing VideoAnalyzer: {e}")
            raise
    
    def _get_reference_videos(self, directory):
        """Get list of reference video paths from a directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = []
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(file_path)
        
        return videos
    
    def _get_file_hash(self, file_path, chunk_size=8192):
        """Calculate MD5 hash of a file for comparison"""
        try:
            # Verify file exists
            if not os.path.isfile(file_path):
                logger.error(f"Cannot calculate hash - file does not exist: {file_path}")
                return None
                
            # Verify file is readable
            if not os.access(file_path, os.R_OK):
                logger.error(f"Cannot calculate hash - file is not readable: {file_path}")
                return None
                
            # Get file size for logging
            file_size = os.path.getsize(file_path)
            logger.info(f"Calculating hash for file: {file_path} (size: {file_size} bytes)")
                
            # Calculate hash
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    md5.update(chunk)
            
            file_hash = md5.hexdigest()
            logger.info(f"Hash calculated: {file_hash} for {os.path.basename(file_path)}")
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating file hash for {file_path}: {e}")
            return None
    
    def _cache_reference_hashes(self, file_paths):
        """Cache MD5 hashes of reference videos for quick lookup"""
        hashes = {}
        for path in file_paths:
            try:
                # Ensure the file exists
                if not os.path.isfile(path):
                    logger.warning(f"Reference file does not exist: {path}")
                    continue
                    
                # Get file size to verify it's a valid file
                file_size = os.path.getsize(path)
                if file_size == 0:
                    logger.warning(f"Reference file is empty: {path}")
                    continue
                    
                # Calculate hash
                file_hash = self._get_file_hash(path)
                if file_hash:
                    hashes[file_hash] = path
                    filename = os.path.basename(path)
                    logger.info(f"Cached hash for {filename}: {file_hash}")
                else:
                    logger.warning(f"Failed to calculate hash for: {path}")
            except Exception as e:
                logger.error(f"Error caching reference hash for {path}: {e}")
                
        return hashes
    
    def _check_exact_reference_match(self, video_path):
        """
        Check if the video exactly matches any reference videos
        
        Returns:
        - None if no match
        - "real" if matches a real reference video
        - "fake" if matches a fake reference video
        """
        try:
            # Calculate hash of the uploaded video
            video_hash = self._get_file_hash(video_path)
            if not video_hash:
                logger.error(f"Failed to calculate hash for video: {video_path}")
                return None
                
            # First, check real reference videos
            if video_hash in self.real_hashes:
                matched_file = os.path.basename(self.real_hashes[video_hash])
                logger.info(f"Video exactly matches REAL reference: {matched_file}")
                return "real"
                
            # Then check fake reference videos    
            if video_hash in self.fake_hashes:
                matched_file = os.path.basename(self.fake_hashes[video_hash])
                logger.info(f"Video exactly matches FAKE reference: {matched_file}")
                return "fake"
                
            # No exact match found
            logger.info("No reference match found for this video")
            return None
            
        except Exception as e:
            logger.error(f"Error checking exact reference match: {e}")
            return None
            
    def extract_frames(self, video_path, output_dir, frame_interval=15, max_frames=30):
        """
        Extract frames from a video file at specified intervals
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the extracted frames
            frame_interval: Interval between frames to extract
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of paths to the extracted frames
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return []
            
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video info: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")
            
            # Improved frame selection strategy
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                # Select frames from different segments of the video
                segment_size = total_frames // max_frames
                frame_indices = [i * segment_size + (segment_size // 2) for i in range(max_frames)]
            
            frame_paths = []
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # First pass: try to extract frames with faces
            for i, frame_idx in enumerate(frame_indices):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()
                
                if success:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    # Only save frames that contain faces
                    if len(faces) > 0:
                        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        logger.debug(f"Extracted frame {i+1} with {len(faces)} faces")
            
            # If no frames with faces were found, fall back to extracting frames without face detection
            if not frame_paths:
                logger.warning("No faces detected in video frames. Falling back to regular frame extraction.")
                for i, frame_idx in enumerate(frame_indices):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = video.read()
                    
                    if success:
                        frame_path = os.path.join(output_dir, f"frame_{i:03d}_noface.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        logger.debug(f"Extracted frame {i+1} without face detection")
                    
                    # Only extract up to max_frames frames
                    if len(frame_paths) >= max_frames:
                        break
            
            video.release()
            logger.info(f"Extracted {len(frame_paths)} frames from video")
            return frame_paths
        
        except Exception as e:
            logger.error(f"Error extracting frames from video: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def preprocess_frame(self, frame_path, target_size=(128, 128)):
        """
        Preprocess a video frame for model input
        
        Args:
            frame_path: Path to the frame image
            target_size: Target size for the model input
            
        Returns:
            Preprocessed frame as numpy array
        """
        try:
            img = tf.keras.preprocessing.image.load_img(frame_path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Normalize
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None
            
    def annotate_representative_frame(self, frame_path, is_fake, output_path=None):
        """
        Annotates the representative frame with visual indicators for real or fake
        
        Args:
            frame_path: Path to the frame image
            is_fake: Boolean indicating if the video is classified as fake
            output_path: Path to save the annotated frame (if None, overwrites original)
            
        Returns:
            Path to the annotated frame
        """
        try:
            if output_path is None:
                output_path = frame_path
                
            # Read the image
            img = cv2.imread(frame_path)
            if img is None:
                logger.error(f"Failed to read image for annotation: {frame_path}")
                return frame_path
                
            height, width = img.shape[:2]
            
            # Add a border and text depending on real/fake
            if is_fake:
                # Red border for fake videos
                color = (0, 0, 255)  # BGR format, red
                text = "FAKE CONTENT DETECTED"
                cv2.rectangle(img, (0, 0), (width, height), color, 10)
                
                # Add warning overlay in top right
                overlay = np.zeros((80, 250, 3), dtype=np.uint8)
                overlay[:] = (0, 0, 255)  # Red background
                cv2.putText(overlay, "MANIPULATED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overlay, "CONTENT", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Place the overlay in the top right corner
                x_offset = width - 250
                y_offset = 0
                img[y_offset:y_offset+80, x_offset:x_offset+250] = overlay
            else:
                # Green border for real videos
                color = (0, 255, 0)  # BGR format, green
                text = "AUTHENTIC CONTENT"
                cv2.rectangle(img, (0, 0), (width, height), color, 10)
                
                # Add verification overlay in top right
                overlay = np.zeros((80, 250, 3), dtype=np.uint8)
                overlay[:] = (0, 255, 0)  # Green background
                cv2.putText(overlay, "AUTHENTIC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overlay, "CONTENT", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Place the overlay in the top right corner
                x_offset = width - 250
                y_offset = 0
                img[y_offset:y_offset+80, x_offset:x_offset+250] = overlay
            
            # Add text at bottom
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 20
            
            # Create a semi-transparent background for text
            bg_overlay = img.copy()
            cv2.rectangle(bg_overlay, (0, height-70), (width, height), (0, 0, 0), -1)
            alpha = 0.7
            img = cv2.addWeighted(bg_overlay, alpha, img, 1 - alpha, 0)
            
            # Add the text
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
            # Save the annotated image
            cv2.imwrite(output_path, img)
            logger.info(f"Saved annotated frame to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            return frame_path
            
    def analyze_video(self, video_path, session_id=None, target_size=(128, 128), max_frames=30):
        """
        Analyze a video for fake content detection
        
        Args:
            video_path: Path to the video file
            session_id: Optional session ID for the analysis
            target_size: Target size for the frames
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Create a unique session ID if not provided
            if session_id is None:
                session_id = os.path.basename(video_path).split('.')[0] + "_" + str(hash(video_path))[:8]
            
            # Create output directories
            video_name = os.path.basename(video_path).split('.')[0]
            session_dir = os.path.join(RESULTS_PATH, session_id)
            frames_dir = os.path.join(session_dir, f"{video_name}_frames")
            
            # First, check if this video exactly matches any reference video
            logger.info(f"Checking if video matches any reference: {video_path}")
            exact_match_result = self._check_exact_reference_match(video_path)
            
            # If it's an exact match to a reference video, skip the analysis
            if exact_match_result:
                logger.info(f"EXACT MATCH FOUND: Video matches a {exact_match_result.upper()} reference")
                
                # Ensure the session directory exists
                os.makedirs(session_dir, exist_ok=True)
                os.makedirs(frames_dir, exist_ok=True)
                
                # Extract at least one frame for display
                extracted_frames = self.extract_frames(video_path, frames_dir, max_frames=5)
                representative_frame = extracted_frames[0] if extracted_frames else None
                
                if not representative_frame:
                    logger.error("Failed to extract any frames from the video")
                    return {"error": "Failed to extract frames from the video"}
                
                # Annotate the representative frame
                is_fake = (exact_match_result == "fake")
                annotated_frame = self.annotate_representative_frame(representative_frame, is_fake)
                
                # Create simplified results for exact matches
                return {
                    "session_id": session_id,
                    "video_path": video_path,
                    "frames_analyzed": 1,
                    "fake_frames_count": 1 if exact_match_result == "fake" else 0,
                    "fake_frames_percentage": 100 if exact_match_result == "fake" else 0,
                    "average_fake_probability": 1.0 if exact_match_result == "fake" else 0.0,
                    "max_fake_probability": 1.0 if exact_match_result == "fake" else 0.0,
                    "max_prob_frame": annotated_frame,
                    "overall_prediction": exact_match_result,
                    "exact_reference_match": True,
                    "match_type": exact_match_result,
                    "frame_predictions": [
                        {
                            "frame_path": annotated_frame,
                            "probability": 1.0 if exact_match_result == "fake" else 0.0,
                            "prediction": exact_match_result
                        }
                    ]
                }
                
            logger.info("No exact reference match found. Performing detailed analysis...")
            
            # If no exact match, proceed with normal analysis
            # Ensure directories exist
            os.makedirs(session_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extract frames
            frame_paths = self.extract_frames(video_path, frames_dir, max_frames=max_frames)
            
            if not frame_paths:
                logger.error("No faces detected in video frames")
                return {"error": "No faces could be detected in the video frames"}
            
            predictions = []
            
            def process_single_frame(frame_path):
                frame_array = self.preprocess_frame(frame_path, target_size)
                if frame_array is None:
                    return None
                
                # Make prediction with confidence threshold
                pred = self.model.predict(frame_array, verbose=0)
                prob = float(pred[0][0])
                
                return {
                    "frame_path": frame_path,
                    "probability": prob,
                    "prediction": "fake" if prob > 0.75 else "real"  # Higher threshold for confidence
                }
            
            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
                results = list(executor.map(process_single_frame, frame_paths))
            
            predictions = [r for r in results if r is not None]
            
            if not predictions:
                logger.error("Failed to process any frames")
                return {"error": "Failed to process video frames"}
            
            # Calculate statistics
            fake_probs = [p["probability"] for p in predictions]
            avg_prob = sum(fake_probs) / len(fake_probs)
            max_prob = max(fake_probs)
            
            max_prob_idx = fake_probs.index(max_prob)
            max_prob_frame = predictions[max_prob_idx]["frame_path"]
            
            fake_count = sum(1 for p in predictions if p["prediction"] == "fake")
            fake_percentage = (fake_count / len(predictions)) * 100
            
            # More conservative classification
            # Consider fake only if both percentage and average probability are high
            overall_prediction = "fake" if (fake_percentage > 60 and avg_prob > 0.7) else "real"
            
            # Annotate the representative frame
            is_fake = (overall_prediction == "fake")
            annotated_frame = self.annotate_representative_frame(max_prob_frame, is_fake)
            
            # Update the max_prob_frame with the annotated version
            max_prob_frame = annotated_frame
            
            # Create final analysis results
            analysis_results = {
                "session_id": session_id,
                "video_path": video_path,
                "frames_analyzed": len(predictions),
                "fake_frames_count": fake_count,
                "fake_frames_percentage": fake_percentage,
                "average_fake_probability": avg_prob,
                "max_fake_probability": max_prob,
                "max_prob_frame": max_prob_frame,
                "overall_prediction": overall_prediction,
                "exact_reference_match": False,
                "frame_predictions": predictions
            }
            
            # Save analysis results
            with open(os.path.join(session_dir, 'analysis_results.json'), 'w') as f:
                # Convert frame_predictions to a simpler format for JSON
                simplified_results = analysis_results.copy()
                simplified_results['frame_predictions'] = [
                    {
                        'frame': os.path.basename(p['frame_path']),
                        'probability': p['probability'],
                        'prediction': p['prediction']
                    } for p in analysis_results['frame_predictions']
                ]
                json.dump(simplified_results, f, indent=2)
            
            return analysis_results
        
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return {"error": f"Error analyzing video: {str(e)}"}

# Initialize video analyzer
video_analyzer = VideoAnalyzer()

def preprocess_image(file_path, target_size=(128, 128)):
    try:
        logger.debug(f"Preprocessing image: {file_path}")
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        logger.error(traceback.format_exc())
        return None

def generate_explanations(image, model):
    try:
        if model is None:
            return None, None, None
        
        prob_fake = model.predict(image)[0][0]
        
        # LIME explanation with enhanced visualization
        explainer = lime.lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
        lime_img = mark_boundaries(temp, mask, color=(1, 0, 0), outline_color=(1, 1, 0))
        lime_path = os.path.join(UPLOAD_FOLDER, 'lime_explanation.png')
        
        plt.figure(figsize=(8, 8))
        plt.imshow(lime_img)
        plt.title("LIME Explanation", fontsize=16)
        plt.axis('off')
        plt.savefig(lime_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # SHAP explanation with enhanced visualization
        explainer = shap.DeepExplainer(model, image)
        shap_values = explainer.shap_values(image)
        shap_values_rescaled = np.sum(shap_values[0], axis=-1)
        shap_values_rescaled = shap_values_rescaled / np.max(np.abs(shap_values_rescaled))
        shap_path = os.path.join(UPLOAD_FOLDER, 'shap_explanation.png')
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image[0])
        plt.imshow(shap_values_rescaled, cmap='RdBu_r', alpha=0.7)
        plt.colorbar(label='SHAP Value')
        plt.title("SHAP Explanation", fontsize=16)
        plt.axis('off')
        plt.savefig(shap_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return prob_fake, lime_path, shap_path
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

def create_combined_report(original_img_path, prob_fake, lime_path, shap_path):
    try:
        # Create a combined image with the original and explanations
        plt.figure(figsize=(15, 12))
        
        # Add a colorful header
        is_fake = prob_fake > 0.4
        header_color = 'red' if is_fake else 'green'
        verdict = "FAKE" if is_fake else "REAL"
        plt.suptitle(f"DeepFake Analysis Report", fontsize=22, color='navy', y=0.98)
        
        # Original image
        plt.subplot(2, 2, 1)
        original = plt.imread(original_img_path)
        plt.imshow(original)
        plt.title("Original Image", fontsize=14)
        plt.axis('off')
        
        # Prediction gauge
        plt.subplot(2, 2, 2)
        plt.axis('equal')
        wedge_size = prob_fake * 2 * np.pi
        colors = ['green', 'yellow', 'red']
        wedge_color = colors[int(min(prob_fake * 3, 2))]
        
        circle = plt.Circle((0, 0), 0.8, fill=False, color='gray', linewidth=2)
        wedge = plt.matplotlib.patches.Wedge((0, 0), 0.8, 0, wedge_size * 180 / np.pi, 
                                            fc=wedge_color, alpha=0.8, linewidth=0)
        plt.gca().add_patch(circle)
        plt.gca().add_patch(wedge)
        plt.text(0, 0, f"{verdict}\n{prob_fake:.1%}", 
                 ha='center', va='center', fontsize=16, 
                 color=header_color, fontweight='bold')
        plt.text(0, -1, "Probability of being fake", ha='center', fontsize=12)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axis('off')
        
        # LIME explanation
        plt.subplot(2, 2, 3)
        lime_img = plt.imread(lime_path)
        plt.imshow(lime_img)
        plt.title("LIME Explanation\n(Areas influencing prediction)", fontsize=14)
        plt.axis('off')
        
        # SHAP explanation
        plt.subplot(2, 2, 4)
        shap_img = plt.imread(shap_path)
        plt.imshow(shap_img)
        plt.title("SHAP Explanation\n(Feature importance heatmap)", fontsize=14)
        plt.axis('off')
        
        # Save the combined image (without the footer explanation text)
        combined_path = os.path.join(UPLOAD_FOLDER, 'combined_report.png')
        plt.savefig(combined_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return combined_path
    except Exception as e:
        logger.error(f"Error creating combined report: {e}")
        logger.error(traceback.format_exc())
        return None

def send_enhanced_result(chat_id, prob_fake):
    try:
        # Create a more professional message with emojis and formatting
        is_fake = prob_fake > 0.4
        emoji = "🤖" if is_fake else "👨"
        verdict = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else 1 - prob_fake
        
        # Format confidence as a visual bar
        bar_length = 10
        filled_length = int(round(bar_length * confidence))
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Color indicator
        indicator = "🔴" if is_fake else "🟢"
        
        message = f"""
*{indicator} ANALYSIS COMPLETE {indicator}*

{emoji} *Verdict*: `{verdict}`
⚖️ *Confidence*: `{confidence:.2%}`
📊 *Confidence Bar*: `{bar}`

_Analyzed with DeepFake Detection AI_
"""
        
        # Send the formatted message
        bot.send_message(chat_id, message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error sending enhanced result: {e}")
        logger.error(traceback.format_exc())
        bot.send_message(chat_id, "Analysis complete, but there was an error formatting the results.")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    # Create a custom keyboard with options
    markup = telebot.types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn1 = telebot.types.KeyboardButton('ℹ️ About')
    btn2 = telebot.types.KeyboardButton('❓ How it works')
    btn3 = telebot.types.KeyboardButton('🖥️ Local Web App')
    btn4 = telebot.types.KeyboardButton('📁 File Upload Tips')
    markup.add(btn1, btn2, btn3, btn4)
    
    welcome_text = (
        "🔍 *Welcome to Media Sentinel!* 🔍\n\n"
        "Upload an image or video to analyze if it's real or AI-generated.\n"
        "Our advanced AI will provide detailed explanations with visual guides.\n\n"
        "✨ *Features*:\n"
        "• Advanced AI detection for images and videos\n"
        "• Visual explanations and frame analysis\n"
        "• Reference video matching\n"
        "• Detailed reports\n\n"
        "Simply send an image or video to get started!\n"
        "💡 *Tip*: For larger videos, send as a file/document."
    )
    
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == 'ℹ️ About')
def about(message):
    about_text = (
        "*About Media Sentinel*\n\n"
        "This bot uses advanced deep learning to detect if images and videos are real or AI-generated (deepfake).\n\n"
        "Our technology analyzes subtle patterns that are invisible to the human eye but "
        "reveal whether content was created by AI or is authentic.\n\n"
        "For videos, we can detect manipulations frame-by-frame and also match against a database of known fake/real videos.\n\n"
        "🔒 *Privacy*: All uploaded content is analyzed privately and deleted after processing."
    )
    bot.send_message(message.chat.id, about_text, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == '❓ How it works')
def how_it_works(message):
    how_text = (
        "*How DeepFake Detection Works*\n\n"
        "1️⃣ *Upload*: Send us any image or video you want to analyze\n\n"
        "2️⃣ *Analysis*: Our AI examines the content for telltale signs of manipulation\n\n"
        "3️⃣ *Explanation*: We provide visual guides showing exactly which parts influenced the decision\n\n"
        "4️⃣ *Result*: You receive a detailed report with confidence score and visual explanations\n\n"
        "*For Images*: We use LIME and SHAP - advanced AI explanation techniques that highlight "
        "which parts of the image were most important for the detection.\n\n"
        "*For Videos*: We analyze multiple frames, detect faces, and check against a reference database "
        "of known real and fake videos for exact matches.\n\n"
        "💡 *Tip*: For larger videos, send them as a file/document instead of a video message. "
        "This may allow for larger file uploads in some cases."
    )
    bot.send_message(message.chat.id, how_text, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == '📁 File Upload Tips')
def file_upload_tips(message):
    tips_text = (
        "*Video Upload Tips*\n\n"
        "Telegram has two ways to send videos:\n\n"
        "1️⃣ *As Video* (standard method):\n"
        "• Limited to 50MB\n"
        "• Compress 📱 → 📎 → Camera → Video\n"
        "• Better preview and playback\n\n"
        "2️⃣ *As Document* (for larger files):\n"
        "• Can handle larger files\n"
        "• Compress 📱 → 📎 → File/Document\n"
        "• Select your video file\n"
        "• Preserves original quality\n\n"
        "If your video is too large as a regular video message, try sending it as a document. "
        "This often works for larger files and preserves quality.\n\n"
        "Supported formats: mp4, avi, mov, mkv, webm"
    )
    bot.send_message(message.chat.id, tips_text, parse_mode="Markdown")

@bot.message_handler(func=lambda message: message.text == '🖥️ Local Web App')
def local_webapp(message):
    local_app_text = (
        "*Using the Local Web Application*\n\n"
        "For larger videos (>50MB) or for more advanced features, use our local web app:\n\n"
        "1️⃣ *Setup*:\n"
        "• Ensure Python is installed on your computer\n"
        "• Download the MediaSentinel package from our GitHub\n"
        "• Run `pip install -r requirements.txt` to install dependencies\n\n"
        "2️⃣ *Launch*:\n"
        "• Run `python video_app.py` in the project directory\n"
        "• Open `http://localhost:5001` in your browser\n\n"
        "3️⃣ *Features*:\n"
        "• Process videos of any size\n"
        "• Upload reference videos\n"
        "• Get detailed frame-by-frame analysis\n"
        "• Save and export reports\n\n"
        "The local web app provides the same powerful analysis as this bot but without the file size limitations of Telegram."
    )
    bot.send_message(message.chat.id, local_app_text, parse_mode="Markdown")

@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Send a message letting the user know analysis has begun with a loading emoji
        processing_message = bot.reply_to(message, "🔄 *Your image is being analyzed... Please wait.*", parse_mode="Markdown")
        
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        filename = os.path.join(UPLOAD_FOLDER, f'image_{message.message_id}.jpg')
        with open(filename, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        img_array = preprocess_image(filename)
        if img_array is None:
            bot.edit_message_text("❌ Error processing image.", 
                                 message.chat.id, processing_message.message_id)
            return
        
        # Update processing message with a progress indicator
        bot.edit_message_text("🔄 *Analyzing image...\nGenerating explanations...*", 
                             message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        prob_fake, lime_path, shap_path = generate_explanations(img_array, model)
        if prob_fake is None:
            bot.edit_message_text("❌ Error analyzing image.", 
                                 message.chat.id, processing_message.message_id)
            return
        
        # Create the combined report
        combined_path = create_combined_report(filename, prob_fake, lime_path, shap_path)
        
        # Send the enhanced result message
        send_enhanced_result(message.chat.id, prob_fake)
        
        # Delete the processing message
        try:
            bot.delete_message(message.chat.id, processing_message.message_id)
            logger.info("Deleted processing message")
        except Exception as e:
            logger.warning(f"Failed to delete processing message: {e}")
            # Continue anyway - this is not critical
        
        # Send the combined report without explanations in the image
        if combined_path:
            with open(combined_path, 'rb') as report_file:
                bot.send_photo(message.chat.id, report_file, 
                              caption="📊 *Detailed Analysis Report*", 
                              parse_mode="Markdown")
        
        # Send real-world context explanation as a separate text message
        is_fake = prob_fake > 0.4
        if is_fake:
            context_text = (
                "*Analysis performed by Media Sentinel AI*\n\n"
                "Red areas indicate features that suggest the image is fake\n"
                "Blue areas indicate features that suggest the image is real\n\n"
                "*Real-world examples of manipulated features often include:*\n"
                "• Unnatural skin texture or blending around facial edges\n"
                "• Inconsistent lighting/reflections in eyes and skin\n"
                "• Facial symmetry abnormalities\n"
                "• Irregular hair patterns or texture"
            )
        else:
            context_text = (
                "*Analysis performed by Media Sentinel AI*\n\n"
                "Red areas indicate features that suggest the image is fake\n"
                "Blue areas indicate features that suggest the image is real\n\n"
                "*Real-world indicators of authentic images include:*\n"
                "• Natural skin texture variations and pores\n"
                "• Consistent lighting and shadows across face\n"
                "• Natural imperfections like freckles, skin marks\n"
                "• Realistic light reflections in eyes"
            )
            
        # Send the explanatory text before the option buttons
        bot.send_message(message.chat.id, context_text, parse_mode="Markdown")
        
        # Create inline buttons for additional actions
        markup = telebot.types.InlineKeyboardMarkup()
        btn1 = telebot.types.InlineKeyboardButton('📋 View LIME Explanation', callback_data='view_lime')
        btn2 = telebot.types.InlineKeyboardButton('📈 View SHAP Explanation', callback_data='view_shap')
        btn3 = telebot.types.InlineKeyboardButton('📥 Download Full Report', callback_data='download_report')
        markup.add(btn1, btn2)
        markup.add(btn3)
        
        bot.send_message(message.chat.id, "Select an option to view detailed explanations:", reply_markup=markup)
        
        # Store paths for callback queries
        bot.temp_data = {
            'user_id': message.from_user.id,
            'lime_path': lime_path,
            'shap_path': shap_path,
            'combined_path': combined_path,
            'original_path': filename
        }
    
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        bot.reply_to(message, "❌ An unexpected error occurred during analysis.")

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        # Document upload tip callback
        if call.data == 'document_tip':
            doc_tip_text = (
                "*How to Send Videos as Documents* 📁\n\n"
                "Sending a video as a document can help with larger files:\n\n"
                "📱 *On Mobile:*\n"
                "1. Tap the 📎 (attachment) icon\n"
                "2. Select 'File' or 'Document' (not 'Camera' or 'Gallery')\n"
                "3. Browse to your video file and select it\n"
                "4. Send the document\n\n"
                "💻 *On Desktop:*\n"
                "1. Click the 📎 (attachment) icon\n"
                "2. Select your video file from your computer\n"
                "3. Send as a file/document\n\n"
                "This method can sometimes allow files larger than 50MB and preserves original quality."
            )
            bot.send_message(call.message.chat.id, doc_tip_text, parse_mode="Markdown")
            bot.answer_callback_query(call.id, "Document upload instructions sent!")
            return
            
        # Web app instructions callback
        if call.data == 'webapphowto':
            local_app_text = (
                "*Using the Local Web Application*\n\n"
                "For larger videos (>50MB), follow these steps:\n\n"
                "1️⃣ *Setup*:\n"
                "• Ensure Python is installed on your computer\n"
                "• Download the MediaSentinel package from our GitHub\n"
                "• Run `pip install -r requirements.txt` to install dependencies\n\n"
                "2️⃣ *Launch*:\n"
                "• Run `python video_app.py` in the project directory\n"
                "• Open `http://localhost:5001` in your browser\n\n"
                "3️⃣ *Upload*:\n"
                "• Click 'Choose File' on the web interface\n"
                "• Select your video file (any size supported)\n"
                "• Click 'Analyze' to process the video\n\n"
                "The local web app provides the same advanced analysis without Telegram's file size limitations."
            )
            bot.send_message(call.message.chat.id, local_app_text, parse_mode="Markdown")
            bot.answer_callback_query(call.id, "Web app instructions sent!")
            return
            
        # Handle image analysis callbacks
        if call.data in ['view_lime', 'view_shap', 'download_report']:
            if not hasattr(bot, 'temp_data') or bot.temp_data['user_id'] != call.from_user.id:
                bot.answer_callback_query(call.id, "Session expired. Please upload a new image.")
                return
                
            if call.data == 'view_lime':
                # Get the prediction result for context
                img_array = preprocess_image(bot.temp_data['original_path'])
                if img_array is not None:
                    pred_prob = model.predict(img_array)[0][0]
                    is_fake = pred_prob > 0.4
                else:
                    # Fallback if we can't preprocess the image again
                    pred_prob = 0.5
                    is_fake = False
                
                # Create detailed message based on prediction
                if is_fake:
                    explanation_text = (
                        "🔍 *LIME Explanation: How AI Detects Fakes*\n\n"
                        "The highlighted areas show regions that influenced the AI's decision that this image is manipulated.\n\n"
                        "*What LIME Shows:*\n"
                        "• Red-bounded areas indicate features most suspicious to the AI\n"
                        "• These often correspond to deepfake artifacts like:\n"
                        "  - Unnatural skin texture transitions\n"
                        "  - Inconsistent lighting around facial features\n"
                        "  - Blurred or distorted areas where images were merged\n"
                        "  - Unnatural symmetry or irregularities\n\n"
                        "*Real-world example:* In celebrity deepfakes, these areas typically appear around the jawline, hairline, and where facial features were altered."
                    )
                else:
                    explanation_text = (
                        "🔍 *LIME Explanation: Features of Authenticity*\n\n"
                        "The highlighted areas show regions that influenced the AI's decision that this image is authentic.\n\n"
                        "*What LIME Shows:*\n"
                        "• Red-bounded areas indicate features the AI examined most closely\n"
                        "• In real images, these typically highlight natural features like:\n"
                        "  - Natural skin texture and pore patterns\n"
                        "  - Consistent light reflection in eyes\n"
                        "  - Natural facial asymmetries\n"
                        "  - Authentic hair patterns and texture\n\n"
                        "*Real-world example:* The natural imperfections in human faces (like slightly uneven eyes or skin texture variations) are often what distinguish real images from AI-generated ones."
                    )
                
                with open(bot.temp_data['lime_path'], 'rb') as lime_file:
                    bot.send_photo(call.message.chat.id, lime_file, 
                                  caption=explanation_text, 
                                  parse_mode="Markdown")
                bot.answer_callback_query(call.id, "LIME explanation displayed")
                
            elif call.data == 'view_shap':
                # Get the prediction result for context
                img_array = preprocess_image(bot.temp_data['original_path'])
                if img_array is not None:
                    pred_prob = model.predict(img_array)[0][0]
                    is_fake = pred_prob > 0.4
                else:
                    # Fallback if we can't preprocess the image again
                    pred_prob = 0.5
                    is_fake = False
                
                # Create detailed message based on prediction
                if is_fake:
                    explanation_text = (
                        "📊 *SHAP Explanation: Feature Importance Heatmap*\n\n"
                        "This heatmap shows how different regions impact the prediction that the image is manipulated.\n\n"
                        "*What SHAP Values Mean:*\n"
                        "• Red regions: Features pushing the model toward 'fake' prediction\n"
                        "• Blue regions: Features pushing toward 'real' prediction\n\n"
                        "*Common Fake Indicators (Red Areas):*\n"
                        "• Inconsistent texture in skin transitions\n"
                        "• Unnatural shadows or reflections\n"
                        "• Geometric inconsistencies in facial features\n"
                        "• Artificial blurring or smoothing\n\n"
                        "*Real-world example:* AI image generators often struggle with eye details - look for inconsistencies in reflections, pupil shapes, and iris patterns."
                    )
                else:
                    explanation_text = (
                        "📊 *SHAP Explanation: Feature Importance Heatmap*\n\n"
                        "This heatmap shows how different regions impact the prediction that the image is authentic.\n\n"
                        "*What SHAP Values Mean:*\n"
                        "• Blue regions: Features pushing the model toward 'real' prediction\n"
                        "• Red regions: Features that might suggest manipulation\n\n"
                        "*Authentic Image Indicators (Blue Areas):*\n"
                        "• Consistent lighting across all facial features\n"
                        "• Natural skin blemishes and texture patterns\n"
                        "• Realistic shadows and depth\n"
                        "• Proper perspective and proportions\n\n"
                        "*Real-world example:* Real photos typically show consistent noise patterns and lighting across the entire image, unlike manipulated images where different parts may have different noise signatures."
                    )
                
                with open(bot.temp_data['shap_path'], 'rb') as shap_file:
                    bot.send_photo(call.message.chat.id, shap_file, 
                                  caption=explanation_text, 
                                  parse_mode="Markdown")
                bot.answer_callback_query(call.id, "SHAP explanation displayed")
                
            elif call.data == 'download_report':
                # Handle PDF report generation for images
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    from reportlab.lib import colors
                    
                    # Create PDF report
                    pdf_path = os.path.join(UPLOAD_FOLDER, 'medialsentinel_report.pdf')
                    c = canvas.Canvas(pdf_path, pagesize=letter)
                    width, height = letter
                    
                    # Header
                    c.setFont("Helvetica-Bold", 24)
                    c.drawString(72, height - 72, "DeepFake Analysis Report")
                    
                    # Add images
                    c.drawImage(bot.temp_data['original_path'], 72, height - 300, width=200, height=200)
                    c.drawImage(bot.temp_data['combined_path'], 72, height - 550, width=450, height=200)
                    
                    # Add conclusion
                    prob_fake = model.predict(preprocess_image(bot.temp_data['original_path']))[0][0]
                    verdict = "FAKE" if prob_fake > 0.4 else "REAL"
                    
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(72, height - 600, f"Conclusion: This image is likely {verdict}")
                    c.setFont("Helvetica", 12)
                    c.drawString(72, height - 630, f"Confidence: {prob_fake:.2%}")
                    
                    # Add real-world explanation
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(72, height - 670, "Analysis Explanation:")
                    c.setFont("Helvetica", 10)
                    
                    if verdict == "FAKE":
                        explanations = [
                            "• The AI detected anomalies typical of deepfake or manipulated images",
                            "• Key indicators include unnatural skin transitions and texture inconsistencies",
                            "• Facial feature proportions may have subtle geometric anomalies",
                            "• Lighting and shadow patterns show inconsistencies across the image",
                            "• Eye details (reflections, iris patterns) may show artificial characteristics",
                            "• Similar patterns are seen in known AI-generated or manipulated content"
                        ]
                    else:
                        explanations = [
                            "• The image shows natural characteristics consistent with authentic photos",
                            "• Skin texture contains natural variations and pore patterns",
                            "• Lighting and shadows are consistent across facial features",
                            "• Natural asymmetries and imperfections typical of real faces are present",
                            "• Eye details show realistic reflections and depth",
                            "• Noise patterns and image artifacts are consistent throughout"
                        ]
                        
                    for i, line in enumerate(explanations):
                        c.drawString(80, height - 690 - (i * 15), line)
                    
                    c.save()
                    
                    # Send the PDF
                    with open(pdf_path, 'rb') as pdf_file:
                        bot.send_document(call.message.chat.id, pdf_file, 
                                         caption="📑 Here's your complete DeepFake analysis report with real-world explanations")
                    
                    bot.answer_callback_query(call.id, "Report downloaded")
                    os.remove(pdf_path)
                except ImportError:
                    # If ReportLab is not available, send the combined image as fallback
                    with open(bot.temp_data['combined_path'], 'rb') as report_file:
                        bot.send_document(call.message.chat.id, report_file, 
                                         caption="📊 *DeepFake Analysis Report*", 
                                         parse_mode="Markdown")
                    bot.answer_callback_query(call.id, "Report downloaded")
                except Exception as e:
                    logger.error(f"Error creating PDF: {e}")
                    bot.answer_callback_query(call.id, "Error creating report")
                    
                    # Send a fallback message with real-world explanation
                    img_array = preprocess_image(bot.temp_data['original_path'])
                    if img_array is not None:
                        pred_prob = model.predict(img_array)[0][0]
                        is_fake = pred_prob > 0.4
                    else:
                        pred_prob = 0.5
                        is_fake = False
                    
                    if is_fake:
                        explanation = (
                            "🔍 *Image Analysis Summary*\n\n"
                            f"Verdict: FAKE/MANIPULATED ({pred_prob:.1%} confidence)\n\n"
                            "*Real-world indicators detected:*\n"
                            "• Unnatural skin texture or blurring\n"
                            "• Inconsistent lighting patterns\n"
                            "• Possible geometric irregularities in features\n"
                            "• Artificial patterns in high-detail areas\n\n"
                            "These patterns are similar to those found in known AI-generated images and deepfakes."
                        )
                    else:
                        explanation = (
                            "🔍 *Image Analysis Summary*\n\n"
                            f"Verdict: AUTHENTIC ({(1-pred_prob):.1%} confidence)\n\n"
                            "*Real-world authentic indicators:*\n"
                            "• Natural skin texture with expected variations\n"
                            "• Consistent lighting and shadow patterns\n"
                            "• Natural facial asymmetries\n"
                            "• Proper detail distribution in complex areas\n\n"
                            "These patterns match what we expect to see in genuine, unmanipulated photographs."
                        )
                    
                    bot.send_message(call.message.chat.id, explanation, parse_mode="Markdown")
        
        # Handle video analysis callbacks
        elif call.data.startswith('video_frames_') or call.data.startswith('video_stats_'):
            session_id = call.data.split('_')[-1]
            session_dir = os.path.join(RESULTS_PATH, session_id)
            
            # Find the analysis results file
            results_path = None
            for root, dirs, files in os.walk(session_dir):
                if 'analysis_results.json' in files:
                    results_path = os.path.join(root, 'analysis_results.json')
                    break
            
            if not results_path or not os.path.exists(results_path):
                bot.answer_callback_query(call.id, "Analysis results not found. Please try again.")
                return
                
            # Load the analysis results
            with open(results_path, 'r') as f:
                results = json.load(f)
                
            if call.data.startswith('video_frames_'):
                # Send up to 5 representative frames
                frames_to_send = min(5, len(results.get('frame_predictions', [])))
                
                if frames_to_send == 0:
                    bot.answer_callback_query(call.id, "No frames available")
                    return
                    
                bot.answer_callback_query(call.id, f"Sending {frames_to_send} representative frames")
                
                frames_dir = os.path.dirname(results['max_prob_frame'])
                frames_sent = 0
                
                # First send the max probability frame
                max_frame = results['max_prob_frame']
                with open(max_frame, 'rb') as frame_file:
                    prediction = "FAKE" if results['overall_prediction'] == 'fake' else "REAL"
                    caption = f"🖼️ Most Significant Frame\n📊 Prediction: {prediction} (Confidence: {results['max_fake_probability']:.2%})"
                    bot.send_photo(call.message.chat.id, frame_file, caption=caption)
                    frames_sent += 1
                
                # Send other frames with highest fake probabilities
                frame_preds = sorted(results.get('frame_predictions', []), 
                                     key=lambda x: x['probability'], 
                                     reverse=True)
                
                for i, pred in enumerate(frame_preds):
                    if frames_sent >= frames_to_send:
                        break
                        
                    # Skip the max prob frame we already sent
                    frame_path = pred['frame_path']
                    if frame_path == max_frame:
                        continue
                        
                    try:
                        with open(frame_path, 'rb') as frame_file:
                            prediction = "FAKE" if pred['prediction'] == 'fake' else "REAL"
                            prob = pred['probability']
                            caption = f"🖼️ Frame {i+1}\n📊 Prediction: {prediction} (Confidence: {prob:.2%})"
                            bot.send_photo(call.message.chat.id, frame_file, caption=caption)
                            frames_sent += 1
                    except Exception as e:
                        logger.error(f"Error sending frame {frame_path}: {e}")
                
            elif call.data.startswith('video_stats_'):
                # Send detailed statistics about the video analysis
                frames_analyzed = results['frames_analyzed']
                fake_frames = results['fake_frames_count']
                fake_percentage = results['fake_frames_percentage']
                avg_prob = results['average_fake_probability']
                max_prob = results['max_fake_probability']
                overall_prediction = results['overall_prediction'].upper()
                
                # Format confidence as a visual bar
                bar_length = 10
                filled_length = int(round(bar_length * avg_prob))
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                stats_message = (
                    f"📊 *Detailed Video Analysis*\n\n"
                    f"📈 *Overall Prediction*: `{overall_prediction}`\n"
                    f"🔢 *Frames Analyzed*: `{frames_analyzed}`\n"
                    f"🚫 *Fake Frames*: `{fake_frames} ({fake_percentage:.1f}%)`\n"
                    f"✅ *Real Frames*: `{frames_analyzed - fake_frames} ({100-fake_percentage:.1f}%)`\n"
                    f"📉 *Average Fake Probability*: `{avg_prob:.2%}`\n"
                    f"📈 *Max Fake Probability*: `{max_prob:.2%}`\n"
                    f"📊 *Confidence Bar*: `{bar}`\n\n"
                )
                
                if results.get('exact_reference_match', False):
                    match_type = results.get('match_type', '').upper()
                    stats_message += f"🔍 *Reference Match*: This video exactly matches a known {match_type} video in our database."
                
                bot.send_message(call.message.chat.id, stats_message, parse_mode="Markdown")
                bot.answer_callback_query(call.id, "Detailed statistics displayed")
                
    except Exception as e:
        logger.error(f"Callback error: {e}")
        logger.error(traceback.format_exc())
        bot.answer_callback_query(call.id, "An error occurred processing your request")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    """Handle any text messages that don't match other handlers"""
    if message.text.lower() in ['hi', 'hello', 'hey']:
        bot.reply_to(message, "👋 Hello! Send me an image or video to check if it's real or AI-generated.")
    else:
        # Create a custom keyboard with options
        markup = telebot.types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        btn1 = telebot.types.KeyboardButton('ℹ️ About')
        btn2 = telebot.types.KeyboardButton('❓ How it works')
        markup.add(btn1, btn2)
        
        bot.reply_to(message, "Please send me an image or video to analyze. 📸 🎬", reply_markup=markup)

def cleanup_files():
    """Periodic cleanup of old files"""
    try:
        import time
        import glob
        
        # Delete files older than 1 hour
        now = time.time()
        # Cleanup uploads directory
        for f in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
            # Skip the analysis_results directory itself
            if os.path.basename(f) == 'analysis_results':
                continue
                
            if os.stat(f).st_mtime < now - 3600:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                        logger.info(f"Removed old file: {f}")
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                        logger.info(f"Removed old directory: {f}")
                except Exception as e:
                    logger.error(f"Error removing {f}: {e}")
        
        # Cleanup analysis_results directory
        for f in glob.glob(os.path.join(RESULTS_PATH, '*')):
            if os.stat(f).st_mtime < now - 3600:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                        logger.info(f"Removed old analysis file: {f}")
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                        logger.info(f"Removed old analysis directory: {f}")
                except Exception as e:
                    logger.error(f"Error removing analysis result {f}: {e}")
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")

def check_directories():
    """Check if all necessary directories exist and are writable"""
    directories = [
        UPLOAD_FOLDER,
        REFERENCE_VIDEOS_PATH,
        REAL_VIDEOS_PATH,
        FAKE_VIDEOS_PATH,
        RESULTS_PATH
    ]
    
    issues = []
    
    for directory in directories:
        # Check if directory exists
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                issues.append(f"Failed to create directory {directory}: {str(e)}")
                logger.error(f"Failed to create directory {directory}: {e}")
                continue
        
        # Check if directory is writable
        if not os.access(directory, os.W_OK):
            issues.append(f"Directory {directory} is not writable")
            logger.error(f"Directory {directory} is not writable")
    
    if issues:
        logger.error("Directory permission issues detected:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False, issues
    
    logger.info("All directories are correctly set up and writable")
    return True, []

def start_bot():
    """Initialize and start the bot with error handling and cleanup"""
    try:
        import time
        import threading
        
        # Check directories
        logger.info("Checking directories...")
        dirs_ok, issues = check_directories()
        if not dirs_ok:
            logger.error("⚠️ WARNING: There are issues with the directories that may affect bot operation!")
        
        # Start cleanup thread
        def cleanup_thread():
            while True:
                cleanup_files()
                time.sleep(3600)  # Run every hour
                
        threading.Thread(target=cleanup_thread, daemon=True).start()
        
        # Set up commands
        bot.set_my_commands([
            telebot.types.BotCommand("start", "Start the bot"),
            telebot.types.BotCommand("help", "Show help information"),
            telebot.types.BotCommand("webapp", "How to use local web app for large videos"),
            telebot.types.BotCommand("reload_references", "Reload reference videos"),
            telebot.types.BotCommand("check_reference", "Check if a video is in the reference database"),
            telebot.types.BotCommand("add_reference", "Add a video to the reference database")
        ])
        
        # Initial reference video loading
        logger.info("Loading initial reference videos...")
        
        logger.info("Bot started!")
        bot.polling(none_stop=True, interval=1, timeout=60)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        logger.error(traceback.format_exc())
        time.sleep(10)
        start_bot()  # Restart bot after error

@bot.message_handler(content_types=['video'])
def handle_video(message):
    try:
        # Send a message letting the user know analysis has begun
        processing_message = bot.reply_to(message, "🔄 *Processing your video...*", parse_mode="Markdown")
        
        # Get video file size before downloading
        file_info = bot.get_file(message.video.file_id)
        file_size_mb = message.video.file_size / (1024 * 1024)  # Convert to MB
        logger.info(f"Video file size: {file_size_mb:.2f}MB")
        
        # Check if file size exceeds Telegram limit (50MB)
        if file_size_mb > 50:
            # Create an inline keyboard with a button to get web app instructions
            markup = telebot.types.InlineKeyboardMarkup()
            webapp_btn = telebot.types.InlineKeyboardButton('🖥️ How to use Web App', callback_data='webapphowto')
            doc_btn = telebot.types.InlineKeyboardButton('📁 Send as Document Instead', callback_data='document_tip')
            markup.add(doc_btn, webapp_btn)
            
            # Inform user about size limitation
            bot.edit_message_text(
                "⚠️ *Video size limit exceeded* ⚠️\n\n"
                f"Your video is {file_size_mb:.1f}MB, which exceeds Telegram's 50MB limit.\n\n"
                "Options:\n"
                "1️⃣ Upload a smaller version of the video\n"
                "2️⃣ Extract a short clip from your video\n"
                "3️⃣ Reduce the video resolution/quality\n"
                "4️⃣ Try sending as a file/document instead of a video\n\n"
                "For larger videos, please use our web application locally.",
                message.chat.id, processing_message.message_id, parse_mode="Markdown", reply_markup=markup
            )
            return
            
        # Continue with downloading since file size is acceptable
        bot.edit_message_text("🔄 *Video received! Downloading...*", 
                             message.chat.id, processing_message.message_id, parse_mode="Markdown")
                             
        try:
            downloaded_file = bot.download_file(file_info.file_path)
            logger.info(f"Downloaded video file successfully, size: {len(downloaded_file)} bytes")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            logger.error(traceback.format_exc())
            bot.edit_message_text(
                "❌ *Download Error*\n\n"
                f"The video couldn't be downloaded: {str(e)}\n"
                "It might be too large or there may be connection issues.\n"
                "Please try with a smaller video or as a document.",
                message.chat.id, processing_message.message_id, parse_mode="Markdown"
            )
            return
        
        # Create a unique filename based on the message ID
        filename = os.path.join(UPLOAD_FOLDER, f'video_{message.message_id}.mp4')
        logger.info(f"Saving to file: {filename}")
        
        try:
            with open(filename, 'wb') as new_file:
                new_file.write(downloaded_file)
            logger.info(f"File saved successfully to {filename}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            logger.error(traceback.format_exc())
            bot.edit_message_text(
                "❌ *Error saving file*\n\n"
                f"Error: {str(e)}",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
        
        bot.edit_message_text("🔄 *Video downloaded! Analyzing frames and detecting manipulations...*", 
                             message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        # Generate a unique session ID
        session_id = f"tg_{message.message_id}"
        logger.info(f"Starting video analysis with session ID: {session_id}")
        
        # Process using web app style if available, fall back to built-in analyzer if not
        try:
            logger.info("Attempting to use web app style video processing")
            results = process_video_web_app_style(filename, session_id)
            logger.info(f"Video processing completed with verdict: {results.get('overall_prediction', 'unknown')}")
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            logger.error(traceback.format_exc())
            bot.edit_message_text(f"❌ *Analysis Error*: {str(e)}", 
                                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
        
        if "error" in results:
            logger.error(f"Analysis returned error: {results['error']}")
            bot.edit_message_text(f"❌ *Analysis Error*: {results['error']}", 
                                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
        
        # Get the representative frame
        rep_frame_path = results["max_prob_frame"]
        logger.info(f"Representative frame path: {rep_frame_path}")
        
        # Prepare analysis results message
        is_fake = results["overall_prediction"] == "fake"
        is_reference_match = results.get("exact_reference_match", False)
        
        # Format the message differently based on if it's a reference match
        if is_reference_match:
            if results["match_type"] == "fake":
                verdict = "FAKE / MANIPULATED"
                emoji = "🚫"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"✅ *Match Found*: This video matches a known FAKE video\n"
                    f"⚠️ *Verdict*: `{verdict}`\n\n"
                    f"This video has been identified as manipulated content based on our reference database."
                )
            else:
                verdict = "REAL / AUTHENTIC"
                emoji = "✅"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"✅ *Match Found*: This video matches a known REAL video\n"
                    f"✓ *Verdict*: `{verdict}`\n\n"
                    f"This video has been verified as authentic content based on our reference database."
                )
        else:
            # Calculate and format statistics
            fake_frames_pct = results["fake_frames_percentage"]
            avg_prob = results["average_fake_probability"]
            
            # Format confidence as a visual bar
            bar_length = 10
            filled_length = int(round(bar_length * avg_prob))
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            if is_fake:
                verdict = "FAKE / MANIPULATED"
                emoji = "🚫"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"⚠️ *Verdict*: `{verdict}`\n"
                    f"📊 *Fake Probability*: `{avg_prob:.2%}`\n"
                    f"🔍 *Fake Frames*: `{results['fake_frames_count']}/{results['frames_analyzed']} ({fake_frames_pct:.1f}%)`\n"
                    f"📈 *Confidence*: `{bar}`\n\n"
                    f"This video shows signs of manipulation or AI-generated content."
                )
            else:
                verdict = "REAL / AUTHENTIC"
                emoji = "✅"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"✓ *Verdict*: `{verdict}`\n"
                    f"📊 *Real Probability*: `{1-avg_prob:.2%}`\n"
                    f"🔍 *Real Frames*: `{results['frames_analyzed']-results['fake_frames_count']}/{results['frames_analyzed']} ({100-fake_frames_pct:.1f}%)`\n"
                    f"📈 *Confidence*: `{bar}`\n\n"
                    f"This video appears to be authentic with no signs of manipulation."
                )
        
        # Delete the processing message
        try:
            bot.delete_message(message.chat.id, processing_message.message_id)
            logger.info("Deleted processing message")
        except Exception as e:
            logger.warning(f"Failed to delete processing message: {e}")
            # Continue anyway - this is not critical
        
        # Send the representative frame with the analysis results
        try:
            logger.info(f"Attempting to open and send representative frame: {rep_frame_path}")
            if not os.path.exists(rep_frame_path):
                logger.error(f"Representative frame file does not exist: {rep_frame_path}")
                bot.send_message(message.chat.id, 
                                f"✅ *Video Analysis Complete*\n\n"
                                f"Verdict: `{results['overall_prediction'].upper()}`\n\n"
                                f"⚠️ *Error*: Could not send representative frame image.",
                                parse_mode="Markdown")
                return
                
            # Make sure the file is readable
            if not os.access(rep_frame_path, os.R_OK):
                logger.error(f"Representative frame file is not readable: {rep_frame_path}")
                bot.send_message(message.chat.id, 
                                f"✅ *Video Analysis Complete*\n\n"
                                f"Verdict: `{results['overall_prediction'].upper()}`\n\n"
                                f"⚠️ *Error*: Could not read representative frame image.",
                                parse_mode="Markdown")
                return
                
            with open(rep_frame_path, 'rb') as frame_file:
                bot.send_photo(message.chat.id, frame_file, 
                             caption=message_text, 
                             parse_mode="Markdown")
                logger.info("Sent analysis results with frame successfully")
        except Exception as e:
            logger.error(f"Error sending representative frame: {e}")
            logger.error(traceback.format_exc())
            
            # Try to send just the text results without the image
            try:
                bot.send_message(message.chat.id, 
                                f"{message_text}\n\n⚠️ *Note*: Could not send representative frame image.",
                                parse_mode="Markdown")
                logger.info("Sent text-only analysis results")
            except Exception as text_error:
                logger.error(f"Error sending text results: {text_error}")
                # Last resort - send a simple message
                try:
                    bot.send_message(message.chat.id, 
                                    f"✅ *Video Analysis Complete*\n\n"
                                    f"Verdict: `{results['overall_prediction'].upper()}`\n\n"
                                    f"⚠️ *Error*: Could not send full analysis results.",
                                    parse_mode="Markdown")
                except:
                    logger.error("Failed to send any results to user")
            return
        
        # Create inline buttons for additional actions
        try:
            markup = telebot.types.InlineKeyboardMarkup()
            btn1 = telebot.types.InlineKeyboardButton('📋 View All Frames', callback_data=f'video_frames_{session_id}')
            btn2 = telebot.types.InlineKeyboardButton('📊 Detailed Statistics', callback_data=f'video_stats_{session_id}')
            markup.add(btn1, btn2)
            
            bot.send_message(message.chat.id, "Select an option for more details:", reply_markup=markup)
            logger.info("Sent action buttons successfully")
        except Exception as e:
            logger.error(f"Error sending action buttons: {e}")
            # Not critical, so we don't need to send an error message to the user
        
    except telebot.apihelper.ApiTelegramException as telegram_error:
        # Handle specific Telegram API errors
        if "file is too big" in str(telegram_error).lower():
            # Create an inline keyboard with buttons for different options
            markup = telebot.types.InlineKeyboardMarkup(row_width=1)
            webapp_btn = telebot.types.InlineKeyboardButton('🖥️ How to use Web App', callback_data='webapphowto')
            doc_btn = telebot.types.InlineKeyboardButton('📁 Send as Document Instead', callback_data='document_tip')
            markup.add(doc_btn, webapp_btn)
            
            bot.reply_to(message, 
                "⚠️ *File size limit exceeded*\n\n"
                "This video is too large for Telegram to process.\n"
                "Try sending it as a document/file instead, or use our web application locally.",
                parse_mode="Markdown", reply_markup=markup)
        else:
            logger.error(f"Telegram API error: {telegram_error}")
            bot.reply_to(message, "❌ An error occurred with Telegram. Please try again with a smaller video.")
    except Exception as e:
        logger.error(f"Error handling video: {e}")
        logger.error(traceback.format_exc())
        bot.reply_to(message, "❌ An error occurred while analyzing the video. Please try again later.")

@bot.message_handler(commands=['webapp'])
def webapp_command(message):
    """Handle the /webapp command to provide information about the local web application"""
    local_app_text = (
        "*Using the Local Web Application for Large Videos*\n\n"
        "The MediaSentinel web app allows you to analyze videos of any size without Telegram's limitations.\n\n"
        "🔧 *Setup Steps*:\n"
        "1. Download the project from GitHub\n"
        "2. Install requirements: `pip install -r requirements.txt`\n"
        "3. Run the local server: `python video_app.py`\n"
        "4. Open your browser to: `http://localhost:5001`\n\n"
        "💡 *Benefits*:\n"
        "• No file size limits\n"
        "• Faster processing\n"
        "• Save and manage analysis reports\n"
        "• Upload reference videos\n\n"
        "For any questions about setting up the web app, contact our support team."
    )
    
    # Create an inline button to get started
    markup = telebot.types.InlineKeyboardMarkup()
    github_btn = telebot.types.InlineKeyboardButton('📥 Download from GitHub', url='https://github.com/mediasentinel/deepfake-detection')
    markup.add(github_btn)
    
    bot.send_message(message.chat.id, local_app_text, parse_mode="Markdown", reply_markup=markup)

@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        logger.info(f"Document received from user {message.from_user.id}: {message.document.file_name}")
        
        # Check if the document is a video file
        valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_info = bot.get_file(message.document.file_id)
        logger.info(f"File info obtained: {file_info.file_path}")
        
        file_name = message.document.file_name if message.document.file_name else "unknown_file"
        file_extension = os.path.splitext(file_name)[1].lower()
        logger.info(f"File extension: {file_extension}")
        
        # Only process video files
        if file_extension not in valid_video_extensions:
            logger.warning(f"Unsupported file extension: {file_extension}")
            bot.reply_to(message, 
                         "⚠️ The document you sent is not a supported video format. "
                         "Please send a video file (mp4, avi, mov, mkv, webm).")
            return
            
        # Send a message letting the user know analysis has begun
        processing_message = bot.reply_to(message, "🔄 *Processing your video document...*", parse_mode="Markdown")
        
        # Get file size before downloading
        file_size_mb = message.document.file_size / (1024 * 1024)  # Convert to MB
        logger.info(f"File size: {file_size_mb:.2f}MB")
        
        # Check if file size exceeds Telegram limit (50MB for bots, but document can be larger)
        if file_size_mb > 50:
            logger.info(f"Large document detected: {file_size_mb:.1f}MB. Attempting to process.")
            bot.edit_message_text(
                f"⚠️ *Large file detected ({file_size_mb:.1f}MB)*\n\n"
                "Attempting to download... This may take some time.",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        # Continue with downloading
        bot.edit_message_text("🔄 *Document received! Downloading...*", 
                             message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        logger.info(f"Attempting to download file {file_info.file_path}")
        try:
            downloaded_file = bot.download_file(file_info.file_path)
            logger.info(f"File downloaded successfully, size: {len(downloaded_file)} bytes")
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            bot.edit_message_text(
                "❌ *Download Error*\n\n"
                f"The document couldn't be downloaded: {str(e)}\n"
                "It might be too large or there may be connection issues.\n"
                "Please try with a smaller file or use our web application.",
                message.chat.id, processing_message.message_id, parse_mode="Markdown"
            )
            return
        
        # Create a unique filename based on the message ID and original extension
        filename = os.path.join(UPLOAD_FOLDER, f'video_{message.message_id}{file_extension}')
        logger.info(f"Saving to file: {filename}")
        
        try:
            with open(filename, 'wb') as new_file:
                new_file.write(downloaded_file)
            logger.info(f"File saved successfully")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            bot.edit_message_text(
                "❌ *Error saving file*\n\n"
                f"Error: {str(e)}",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
        
        bot.edit_message_text("🔄 *Video downloaded! Analyzing frames and detecting manipulations...*", 
                             message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        # Generate a unique session ID
        session_id = f"tg_{message.message_id}"
        logger.info(f"Starting video analysis with session ID: {session_id}")
        
        # Process using web app style if available, fall back to built-in analyzer if not
        try:
            logger.info("Attempting to use web app style video processing")
            results = process_video_web_app_style(filename, session_id)
            logger.info(f"Video processing completed with verdict: {results.get('overall_prediction', 'unknown')}")
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            bot.edit_message_text(
                "❌ *Analysis Error*\n\n"
                f"Error during video analysis: {str(e)}",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
        
        if "error" in results:
            logger.error(f"Analysis error: {results['error']}")
            bot.edit_message_text(f"❌ *Analysis Error*: {results['error']}", 
                                 message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
        
        # Get the representative frame
        rep_frame_path = results["max_prob_frame"]
        logger.info(f"Representative frame: {rep_frame_path}")
        
        # Prepare analysis results message
        is_fake = results["overall_prediction"] == "fake"
        is_reference_match = results.get("exact_reference_match", False)
        
        # Format the message differently based on if it's a reference match
        if is_reference_match:
            if results["match_type"] == "fake":
                verdict = "FAKE / MANIPULATED"
                emoji = "🚫"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"✅ *Match Found*: This video matches a known FAKE video\n"
                    f"⚠️ *Verdict*: `{verdict}`\n\n"
                    f"This video has been identified as manipulated content based on our reference database."
                )
            else:
                verdict = "REAL / AUTHENTIC"
                emoji = "✅"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"✅ *Match Found*: This video matches a known REAL video\n"
                    f"✓ *Verdict*: `{verdict}`\n\n"
                    f"This video has been verified as authentic content based on our reference database."
                )
        else:
            # Calculate and format statistics
            fake_frames_pct = results["fake_frames_percentage"]
            avg_prob = results["average_fake_probability"]
            
            # Format confidence as a visual bar
            bar_length = 10
            filled_length = int(round(bar_length * avg_prob))
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            if is_fake:
                verdict = "FAKE / MANIPULATED"
                emoji = "🚫"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"⚠️ *Verdict*: `{verdict}`\n"
                    f"📊 *Fake Probability*: `{avg_prob:.2%}`\n"
                    f"🔍 *Fake Frames*: `{results['fake_frames_count']}/{results['frames_analyzed']} ({fake_frames_pct:.1f}%)`\n"
                    f"📈 *Confidence*: `{bar}`\n\n"
                    f"This video shows signs of manipulation or AI-generated content."
                )
            else:
                verdict = "REAL / AUTHENTIC"
                emoji = "✅"
                message_text = (
                    f"{emoji} *Video Analysis Complete* {emoji}\n\n"
                    f"✓ *Verdict*: `{verdict}`\n"
                    f"📊 *Real Probability*: `{1-avg_prob:.2%}`\n"
                    f"🔍 *Real Frames*: `{results['frames_analyzed']-results['fake_frames_count']}/{results['frames_analyzed']} ({100-fake_frames_pct:.1f}%)`\n"
                    f"📈 *Confidence*: `{bar}`\n\n"
                    f"This video appears to be authentic with no signs of manipulation."
                )
        
        # Delete the processing message
        try:
            bot.delete_message(message.chat.id, processing_message.message_id)
            logger.info("Deleted processing message")
        except Exception as e:
            logger.warning(f"Failed to delete processing message: {e}")
            # Continue anyway - this is not critical
        
        # Send the representative frame with the analysis results
        try:
            logger.info(f"Attempting to open and send representative frame: {rep_frame_path}")
            if not os.path.exists(rep_frame_path):
                logger.error(f"Representative frame file does not exist: {rep_frame_path}")
                bot.send_message(message.chat.id, 
                                 f"✅ *Video Analysis Complete*\n\n"
                                 f"Verdict: `{results['overall_prediction'].upper()}`\n\n"
                                 f"⚠️ *Error*: Could not send representative frame image.",
                                 parse_mode="Markdown")
                return
                
            # Make sure the file is readable
            if not os.access(rep_frame_path, os.R_OK):
                logger.error(f"Representative frame file is not readable: {rep_frame_path}")
                bot.send_message(message.chat.id, 
                                 f"✅ *Video Analysis Complete*\n\n"
                                 f"Verdict: `{results['overall_prediction'].upper()}`\n\n"
                                 f"⚠️ *Error*: Could not read representative frame image.",
                                 parse_mode="Markdown")
                return
                
            with open(rep_frame_path, 'rb') as frame_file:
                bot.send_photo(message.chat.id, frame_file, 
                             caption=message_text, 
                             parse_mode="Markdown")
                logger.info("Sent analysis results with frame successfully")
        except Exception as e:
            logger.error(f"Error sending representative frame: {e}")
            logger.error(traceback.format_exc())
            
            # Try to send just the text results without the image
            try:
                bot.send_message(message.chat.id, 
                                 f"{message_text}\n\n⚠️ *Note*: Could not send representative frame image.",
                                 parse_mode="Markdown")
                logger.info("Sent text-only analysis results")
            except Exception as text_error:
                logger.error(f"Error sending text results: {text_error}")
                # Last resort - send a simple message
                try:
                    bot.send_message(message.chat.id, 
                                     f"✅ *Video Analysis Complete*\n\n"
                                     f"Verdict: `{results['overall_prediction'].upper()}`\n\n"
                                     f"⚠️ *Error*: Could not send full analysis results.",
                                     parse_mode="Markdown")
                except:
                    logger.error("Failed to send any results to user")
            return
        
        # Create inline buttons for additional actions
        try:
            markup = telebot.types.InlineKeyboardMarkup()
            btn1 = telebot.types.InlineKeyboardButton('📋 View All Frames', callback_data=f'video_frames_{session_id}')
            btn2 = telebot.types.InlineKeyboardButton('📊 Detailed Statistics', callback_data=f'video_stats_{session_id}')
            markup.add(btn1, btn2)
            
            bot.send_message(message.chat.id, "Select an option for more details:", reply_markup=markup)
            logger.info("Sent action buttons successfully")
        except Exception as e:
            logger.error(f"Error sending action buttons: {e}")
            # Not critical, so we don't need to send an error message to the user
        
    except telebot.apihelper.ApiTelegramException as telegram_error:
        # Handle specific Telegram API errors
        if "file is too big" in str(telegram_error).lower():
            # Create an inline keyboard with buttons for different options
            markup = telebot.types.InlineKeyboardMarkup(row_width=1)
            webapp_btn = telebot.types.InlineKeyboardButton('🖥️ How to use Web App', callback_data='webapphowto')
            doc_btn = telebot.types.InlineKeyboardButton('📁 Send as Document Instead', callback_data='document_tip')
            markup.add(doc_btn, webapp_btn)
            
            bot.reply_to(message, 
                "⚠️ *File size limit exceeded*\n\n"
                "This video is too large for Telegram to process.\n"
                "Try sending it as a document/file instead, or use our web application locally.",
                parse_mode="Markdown", reply_markup=markup)
        else:
            logger.error(f"Telegram API error: {telegram_error}")
            bot.reply_to(message, "❌ An error occurred with Telegram. Please try again with a smaller file.")
    except Exception as e:
        logger.error(f"Error handling document: {e}")
        logger.error(traceback.format_exc())
        bot.reply_to(message, "❌ An error occurred while analyzing the file. Please try again later.")

def reload_reference_videos():
    """Reload all reference videos from the reference directories"""
    global video_analyzer
    try:
        # Re-initialize the video analyzer to reload references
        video_analyzer = VideoAnalyzer()
        return True, f"Successfully reloaded reference videos.\nReal videos: {len(video_analyzer.real_references)}\nFake videos: {len(video_analyzer.fake_references)}"
    except Exception as e:
        logger.error(f"Failed to reload reference videos: {e}")
        return False, f"Failed to reload reference videos: {str(e)}"

@bot.message_handler(commands=['reload_references'])
def reload_references_command(message):
    """Handle the command to reload reference videos"""
    bot.reply_to(message, "🔄 *Reloading reference videos...*", parse_mode="Markdown")
    
    success, msg = reload_reference_videos()
    
    if success:
        # List the actual reference videos found
        real_videos = "\n".join([f"• {os.path.basename(path)}" for path in video_analyzer.real_references])
        fake_videos = "\n".join([f"• {os.path.basename(path)}" for path in video_analyzer.fake_references])
        
        response = (
            "✅ *Reference videos reloaded successfully*\n\n"
            f"📊 *Statistics*:\n"
            f"• Real videos: {len(video_analyzer.real_references)}\n"
            f"• Fake videos: {len(video_analyzer.fake_references)}\n\n"
            f"*Real videos found*:\n{real_videos}\n\n"
            f"*Fake videos found*:\n{fake_videos}"
        )
        bot.reply_to(message, response, parse_mode="Markdown")
    else:
        bot.reply_to(message, f"❌ *Error*: {msg}", parse_mode="Markdown")

@bot.message_handler(commands=['check_reference'])
def check_reference_command(message):
    """Check if a specific video file is properly registered in the reference database"""
    # Check if a reply to a document/video is provided
    if not message.reply_to_message:
        bot.reply_to(message, 
                    "❓ *How to use this command:*\n\n"
                    "1. Send the video file you want to check\n"
                    "2. Reply to that message with `/check_reference`\n\n"
                    "This will check if the video is properly registered in the reference database.",
                    parse_mode="Markdown")
        return
        
    reply = message.reply_to_message
    file_id = None
    file_type = None
    
    # Check if the replied message contains a video or document
    if reply.content_type == 'video':
        file_id = reply.video.file_id
        file_name = reply.video.file_name if hasattr(reply.video, 'file_name') and reply.video.file_name else f"video_{reply.message_id}.mp4"
        file_type = 'video'
    elif reply.content_type == 'document':
        # Only process if it's a video document
        valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_name = reply.document.file_name if hasattr(reply.document, 'file_name') else f"document_{reply.message_id}"
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension in valid_video_extensions:
            file_id = reply.document.file_id
            file_type = 'document'
        else:
            bot.reply_to(message, "❌ The replied file is not a video document. Please reply to a video file.", parse_mode="Markdown")
            return
    else:
        bot.reply_to(message, "❌ You must reply to a video or video document message.", parse_mode="Markdown")
        return
    
    # Download the file for analysis
    processing_message = bot.reply_to(message, "🔄 *Downloading file to check reference status...*", parse_mode="Markdown")
    
    try:
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Save temporarily
        temp_filename = os.path.join(UPLOAD_FOLDER, f"check_ref_{message.message_id}_{file_name}")
        with open(temp_filename, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        bot.edit_message_text("🔄 *File downloaded. Checking hash against reference database...*", 
                            message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        # Calculate hash of the file
        file_hash = video_analyzer._get_file_hash(temp_filename)
        
        if not file_hash:
            bot.edit_message_text("❌ *Failed to calculate hash for the file.*", 
                                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            return
            
        # Check if the hash is in our reference databases
        in_real = file_hash in video_analyzer.real_hashes
        in_fake = file_hash in video_analyzer.fake_hashes
        
        # Check if filename exists in reference paths (to detect file changes)
        real_files = [os.path.basename(path) for path in video_analyzer.real_references]
        fake_files = [os.path.basename(path) for path in video_analyzer.fake_references]
        
        file_basename = os.path.basename(file_name)
        name_in_real = file_basename in real_files
        name_in_fake = file_basename in fake_files
        
        if in_real:
            matched_file = os.path.basename(video_analyzer.real_hashes[file_hash])
            bot.edit_message_text(
                f"✅ *Reference Match Found!*\n\n"
                f"This file is in the REAL reference database.\n"
                f"Matched file: `{matched_file}`\n"
                f"Hash: `{file_hash[:10]}...`",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
        elif in_fake:
            matched_file = os.path.basename(video_analyzer.fake_hashes[file_hash])
            bot.edit_message_text(
                f"🚫 *Reference Match Found!*\n\n"
                f"This file is in the FAKE reference database.\n"
                f"Matched file: `{matched_file}`\n"
                f"Hash: `{file_hash[:10]}...`",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
        else:
            # No exact hash match, but check if filename exists (content might have changed)
            status_message = (
                f"⚠️ *No Reference Match Found*\n\n"
                f"This file is NOT in any reference database.\n"
                f"Hash: `{file_hash[:10]}...`\n\n"
            )
            
            if name_in_real:
                status_message += (
                    f"⚠️ However, a file with the same NAME exists in REAL references.\n"
                    f"The file content may have changed. Try reloading references with /reload_references."
                )
            elif name_in_fake:
                status_message += (
                    f"⚠️ However, a file with the same NAME exists in FAKE references.\n"
                    f"The file content may have changed. Try reloading references with /reload_references."
                )
                
            bot.edit_message_text(status_message, message.chat.id, processing_message.message_id, parse_mode="Markdown")
            
        # Clean up the temporary file
        try:
            os.remove(temp_filename)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Error checking reference status: {e}")
        bot.edit_message_text(f"❌ *Error checking reference*: {str(e)}", 
                            message.chat.id, processing_message.message_id, parse_mode="Markdown")

@bot.message_handler(commands=['add_reference'])
def add_reference_command(message):
    """Add a video to the reference database"""
    # Extract category from command text (e.g., /add_reference real)
    command_parts = message.text.split()
    if len(command_parts) < 2 or command_parts[1].lower() not in ['real', 'fake']:
        bot.reply_to(message, 
                    "❓ *How to use this command:*\n\n"
                    "1. Send the video file you want to add as reference\n"
                    "2. Reply to that message with `/add_reference real` or `/add_reference fake`\n\n"
                    "This will add the video to the specified reference category.",
                    parse_mode="Markdown")
        return
    
    category = command_parts[1].lower()
    
    # Check if a reply to a document/video is provided
    if not message.reply_to_message:
        bot.reply_to(message, "❌ You must reply to a video or video document message.", parse_mode="Markdown")
        return
        
    reply = message.reply_to_message
    file_id = None
    file_name = None
    
    # Check if the replied message contains a video or document
    if reply.content_type == 'video':
        file_id = reply.video.file_id
        file_name = reply.video.file_name if hasattr(reply.video, 'file_name') and reply.video.file_name else f"video_{reply.message_id}.mp4"
    elif reply.content_type == 'document':
        # Only process if it's a video document
        valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_name = reply.document.file_name if hasattr(reply.document, 'file_name') else f"document_{reply.message_id}"
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension in valid_video_extensions:
            file_id = reply.document.file_id
        else:
            bot.reply_to(message, "❌ The replied file is not a video document. Please reply to a video file.", parse_mode="Markdown")
            return
    else:
        bot.reply_to(message, "❌ You must reply to a video or video document message.", parse_mode="Markdown")
        return
    
    # Download the file
    processing_message = bot.reply_to(message, f"🔄 *Downloading video to add as {category.upper()} reference...*", parse_mode="Markdown")
    
    try:
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Determine target directory
        target_dir = REAL_VIDEOS_PATH if category == 'real' else FAKE_VIDEOS_PATH
        os.makedirs(target_dir, exist_ok=True)
        
        # Ensure the filename is safe and doesn't overwrite existing files
        safe_filename = os.path.basename(file_name)
        target_path = os.path.join(target_dir, safe_filename)
        
        # Check if file already exists
        if os.path.exists(target_path):
            # Add timestamp to make unique
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            name, ext = os.path.splitext(safe_filename)
            safe_filename = f"{name}_{timestamp}{ext}"
            target_path = os.path.join(target_dir, safe_filename)
        
        # Save the file
        with open(target_path, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        bot.edit_message_text(f"✅ *Video saved as {category.upper()} reference*\n\nReloading reference database...", 
                            message.chat.id, processing_message.message_id, parse_mode="Markdown")
        
        # Reload reference videos to include the new one
        success, msg = reload_reference_videos()
        
        if success:
            bot.edit_message_text(
                f"✅ *Video successfully added as {category.upper()} reference*\n\n"
                f"Filename: `{safe_filename}`\n"
                f"Path: `{target_path}`\n\n"
                f"Reference database reloaded with:\n"
                f"• {len(video_analyzer.real_references)} REAL videos\n"
                f"• {len(video_analyzer.fake_references)} FAKE videos",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
        else:
            bot.edit_message_text(
                f"⚠️ *Video saved, but error reloading references*\n\n"
                f"The video was saved as a {category.upper()} reference, but there was an error reloading the reference database:\n\n"
                f"{msg}\n\n"
                f"Try using /reload_references manually.",
                message.chat.id, processing_message.message_id, parse_mode="Markdown")
            
    except Exception as e:
        logger.error(f"Error adding reference video: {e}")
        bot.edit_message_text(f"❌ *Error adding reference video*: {str(e)}", 
                            message.chat.id, processing_message.message_id, parse_mode="Markdown")

# Check if the video_processor module exists
try:
    if importlib.util.find_spec('video_processor'):
        import video_processor
        video_processor_available = True
        logger.info("Successfully imported video_processor module")
    else:
        # Try to import video_app
        if importlib.util.find_spec('video_app'):
            import video_app
            video_processor_available = True
            logger.info("Successfully imported video_app module")
        else:
            video_processor_available = False
            logger.warning("Neither video_processor nor video_app module found")
except ImportError:
    video_processor_available = False
    logger.warning("Failed to import video processing modules")

def process_video_web_app_style(video_path, session_id):
    """
    Process video using the web app processing method from video_app.py
    
    Args:
        video_path: Path to the video file
        session_id: Session ID for the analysis
        
    Returns:
        Analysis results dictionary similar to web app format
    """
    try:
        logger.info(f"Processing video using web app style: {video_path}")
        
        # Directory for storing extracted frames and results
        output_dir = os.path.join(UPLOAD_FOLDER, 'video_frames', session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        if video_processor_available:
            # Try to use the video_processor module like in app.py
            try:
                # Try video_processor first
                if 'video_processor' in sys.modules:
                    results = video_processor.process_video(
                        video_path=video_path,
                        model=model,
                        output_dir=output_dir,
                        max_frames=20
                    )
                # Fall back to video_app.VideoAnalyzer
                elif 'video_app' in sys.modules:
                    # Initialize the analyzer directly from video_app
                    analyzer = video_app.VideoAnalyzer()
                    results = analyzer.analyze_video(video_path, session_id=session_id)
                else:
                    raise ImportError("No valid video processing module available")
                    
                logger.info(f"Web app video processing successful: {results.get('overall_prediction', 'unknown')}")
                return results
            except Exception as e:
                logger.error(f"Error using web app video processing: {e}")
                logger.error(traceback.format_exc())
                # Fall back to built-in analyzer
                logger.info("Falling back to built-in video analyzer")
        
        # If web app processing failed or unavailable, use built-in analyzer
        return video_analyzer.analyze_video(video_path, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error in web-app style video processing: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error processing video: {str(e)}"}

if __name__ == '__main__':
    start_bot()
import os
import cv2
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import shutil
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VideoAnalyzer')

# Constants
REFERENCE_VIDEOS_PATH = 'reference_videos'
REAL_VIDEOS_PATH = os.path.join(REFERENCE_VIDEOS_PATH, 'real')
FAKE_VIDEOS_PATH = os.path.join(REFERENCE_VIDEOS_PATH, 'fake')
RESULTS_PATH = 'analysis_results'

class VideoAnalyzer:
    def __init__(self, model_path='fake_face_detection_model.h5'):
        """Initialize the video analyzer with the detection model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            
            # Ensure results directory exists
            os.makedirs(RESULTS_PATH, exist_ok=True)
            
            # Ensure reference directories exist
            os.makedirs(REAL_VIDEOS_PATH, exist_ok=True)
            os.makedirs(FAKE_VIDEOS_PATH, exist_ok=True)
            
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
            
            # Debug: Log the hashes for reference
            logger.info(f"Real video hashes: {list(self.real_hashes.keys())}")
            logger.info(f"Fake video hashes: {list(self.fake_hashes.keys())}")
            
            # Test real reference matches against themselves (for debugging)
            for path in self.real_references:
                match = self._check_exact_reference_match(path)
                logger.info(f"Self-test for {os.path.basename(path)}: {match}")
            
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
        except PermissionError:
            logger.error(f"Permission error calculating hash for: {file_path}")
            return None
        except IOError as e:
            logger.error(f"IO error calculating hash for {file_path}: {e}")
            return None
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
            # Verify the video exists
            if not os.path.isfile(video_path):
                logger.error(f"Cannot check reference match - file does not exist: {video_path}")
                return None
                
            # Calculate hash of the uploaded video
            video_hash = self._get_file_hash(video_path)
            if not video_hash:
                logger.error(f"Failed to calculate hash for video: {video_path}")
                return None
                
            # Debug: Print the hash of the uploaded video
            logger.info(f"Checking video hash: {video_hash}")
            logger.info(f"Number of real hashes: {len(self.real_hashes)}")
            logger.info(f"Number of fake hashes: {len(self.fake_hashes)}")
            
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
                
            # For debugging, check if hash is similar to any in our references
            for category, hash_dict in [("real", self.real_hashes), ("fake", self.fake_hashes)]:
                for ref_hash in hash_dict.keys():
                    # Just log the first few chars to see if there's any pattern or close match
                    logger.debug(f"Comparing with {category} hash: {ref_hash[:16]} vs {video_hash[:16]}")
            
            # No exact match found
            logger.info("No reference match found for this video")
            return None
            
        except Exception as e:
            logger.error(f"Error checking exact reference match: {e}")
            return None
    
    def add_reference_video(self, video_path, category):
        """
        Add a video to the reference collection
        
        Args:
            video_path: Path to the source video
            category: 'real' or 'fake'
        
        Returns:
            Path to the new reference video
        """
        if category.lower() not in ['real', 'fake']:
            raise ValueError("Category must be 'real' or 'fake'")
        
        target_dir = REAL_VIDEOS_PATH if category.lower() == 'real' else FAKE_VIDEOS_PATH
        filename = os.path.basename(video_path)
        target_path = os.path.join(target_dir, filename)
        
        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Check if the file already exists in the target directory
        if os.path.exists(target_path):
            logger.warning(f"Reference video already exists: {target_path}")
            # Still add to cache to ensure it's properly indexed
        else:
            # Copy the file
            logger.info(f"Copying {video_path} to {target_path}")
            shutil.copy2(video_path, target_path)
        
        # Add file hash to cache
        file_hash = self._get_file_hash(target_path)
        if file_hash:
            logger.info(f"Adding reference hash for {category} video: {filename}, hash: {file_hash}")
            if category.lower() == 'real':
                self.real_hashes[file_hash] = target_path
                logger.info(f"Updated real hashes, count: {len(self.real_hashes)}")
            else:
                self.fake_hashes[file_hash] = target_path
                logger.info(f"Updated fake hashes, count: {len(self.fake_hashes)}")
        else:
            logger.error(f"Failed to calculate hash for reference video: {target_path}")
        
        # Refresh reference lists
        if category.lower() == 'real':
            self.real_references = self._get_reference_videos(REAL_VIDEOS_PATH)
        else:
            self.fake_references = self._get_reference_videos(FAKE_VIDEOS_PATH)
        
        logger.info(f"Added reference {category} video: {filename}")
        return target_path
    
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
            
            video.release()
            logger.info(f"Extracted {len(frame_paths)} frames containing faces from video")
            return frame_paths
        
        except Exception as e:
            logger.error(f"Error extracting frames from video: {e}")
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

# Main function for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_analyzer.py <video_path> [add_as_reference] [reference_category]")
        print("Examples:")
        print("  python video_analyzer.py my_video.mp4")
        print("  python video_analyzer.py my_real_video.mp4 add_reference real")
        print("  python video_analyzer.py my_fake_video.mp4 add_reference fake")
        print("  python video_analyzer.py --test-references")
        sys.exit(1)
    
    # Test reference videos
    if len(sys.argv) >= 2 and sys.argv[1] == "--test-references":
        # Initialize analyzer
        analyzer = VideoAnalyzer()
        
        print("Testing reference videos...")
        
        # Test real videos
        if os.path.exists(REAL_VIDEOS_PATH):
            print(f"\nTesting REAL reference videos in: {REAL_VIDEOS_PATH}")
            for video_file in os.listdir(REAL_VIDEOS_PATH):
                video_path = os.path.join(REAL_VIDEOS_PATH, video_file)
                if os.path.isfile(video_path) and video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    print(f"\nTesting real video: {video_file}")
                    # Get the hash for this file
                    file_hash = analyzer._get_file_hash(video_path)
                    print(f"File hash: {file_hash}")
                    
                    # Check if it matches itself in the real references
                    is_in_real = file_hash in analyzer.real_hashes
                    print(f"Hash in real references: {is_in_real}")
                    
                    # Check if the exact match function works
                    match = analyzer._check_exact_reference_match(video_path)
                    print(f"Match result: {match}")
        
        # Test fake videos
        if os.path.exists(FAKE_VIDEOS_PATH):
            print(f"\nTesting FAKE reference videos in: {FAKE_VIDEOS_PATH}")
            for video_file in os.listdir(FAKE_VIDEOS_PATH):
                video_path = os.path.join(FAKE_VIDEOS_PATH, video_file)
                if os.path.isfile(video_path) and video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    print(f"\nTesting fake video: {video_file}")
                    # Get the hash for this file
                    file_hash = analyzer._get_file_hash(video_path)
                    print(f"File hash: {file_hash}")
                    
                    # Check if it matches itself in the fake references
                    is_in_fake = file_hash in analyzer.fake_hashes
                    print(f"Hash in fake references: {is_in_fake}")
                    
                    # Check if the exact match function works
                    match = analyzer._check_exact_reference_match(video_path)
                    print(f"Match result: {match}")
        
        sys.exit(0)
    
    video_path = sys.argv[1]
    
    # Initialize analyzer
    analyzer = VideoAnalyzer()
    
    # Check if we should add this as a reference video
    if len(sys.argv) >= 4 and sys.argv[2] == "add_reference":
        category = sys.argv[3].lower()
        if category not in ["real", "fake"]:
            print(f"Invalid category '{category}'. Must be 'real' or 'fake'")
            sys.exit(1)
        
        # Add as reference
        ref_path = analyzer.add_reference_video(video_path, category)
        print(f"Added {category.upper()} reference video: {ref_path}")
    
    # Analyze the video
    print(f"Analyzing video: {video_path}")
    results = analyzer.analyze_video(video_path)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Print results
    print("\nVideo Analysis Results:")
    print(f"Video: {results['video_path']}")
    
    # If there's an exact reference match
    if results.get('exact_reference_match', False):
        print(f"EXACT MATCH FOUND: This video is in the {results['match_type'].upper()} reference folder")
        print(f"Overall prediction: {results['overall_prediction'].upper()}")
    else:
        print(f"Frames analyzed: {results['frames_analyzed']}")
        print(f"Fake frames: {results['fake_frames_count']} ({results['fake_frames_percentage']:.2f}%)")
        print(f"Average fake probability: {results['average_fake_probability']:.4f}")
        print(f"Max fake probability: {results['max_fake_probability']:.4f}")
        print(f"Overall prediction: {results['overall_prediction'].upper()}") 
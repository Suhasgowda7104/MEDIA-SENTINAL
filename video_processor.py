import os
import cv2
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VideoProcessor')

def extract_frames(video_path, output_dir, frame_interval=15, max_frames=30):
    """
    Extract frames from a video file at specified intervals with improved frame selection
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

def preprocess_frame(frame_path, target_size=(128, 128)):
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

def process_video(video_path, model, output_dir, target_size=(128, 128), max_frames=30):
    try:
        video_name = os.path.basename(video_path).split('.')[0]
        frames_dir = os.path.join(output_dir, f"{video_name}_frames")
        
        frame_paths = extract_frames(video_path, frames_dir, max_frames=max_frames)
        
        if not frame_paths:
            logger.error("No faces detected in video frames")
            return {"error": "No faces could be detected in the video frames"}
        
        predictions = []
        
        def process_single_frame(frame_path):
            frame_array = preprocess_frame(frame_path, target_size)
            if frame_array is None:
                return None
            
            # Make prediction with confidence threshold
            pred = model.predict(frame_array, verbose=0)
            prob = float(pred[0][0])
            
            return {
                "frame_path": frame_path,
                "probability": prob,
                "prediction": "fake" if prob > 0.8 else "real"  # Increased threshold
            }
        
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            results = list(executor.map(process_single_frame, frame_paths))
        
        predictions = [r for r in results if r is not None]
        
        if not predictions:
            logger.error("Failed to process any frames")
            return {"error": "Failed to process video frames"}
        
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
        
        return {
            "video_path": video_path,
            "frames_analyzed": len(predictions),
            "fake_frames_count": fake_count,
            "fake_frames_percentage": fake_percentage,
            "average_fake_probability": avg_prob,
            "max_fake_probability": max_prob,
            "max_prob_frame": max_prob_frame,
            "overall_prediction": overall_prediction,
            "frame_predictions": predictions
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {"error": f"Error processing video: {str(e)}"}

if __name__ == "__main__":
    # Test the video processor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = "test_output"
    
    # Load model
    model_path = "fake_face_detection_model.h5"
    model = tf.keras.models.load_model(model_path)
    
    # Process video
    results = process_video(video_path, model, output_dir)
    
    # Print results
    print("\nVideo Analysis Results:")
    print(f"Video: {results['video_path']}")
    print(f"Frames analyzed: {results['frames_analyzed']}")
    print(f"Fake frames: {results['fake_frames_count']} ({results['fake_frames_percentage']:.2f}%)")
    print(f"Average fake probability: {results['average_fake_probability']:.4f}")
    print(f"Max fake probability: {results['max_fake_probability']:.4f}")
    print(f"Overall prediction: {results['overall_prediction'].upper()}") 
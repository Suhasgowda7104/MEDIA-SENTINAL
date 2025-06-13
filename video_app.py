import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from video_analyzer import VideoAnalyzer

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure reference videos paths
REFERENCE_VIDEOS_PATH = 'reference_videos'
REAL_VIDEOS_PATH = os.path.join(REFERENCE_VIDEOS_PATH, 'real')
FAKE_VIDEOS_PATH = os.path.join(REFERENCE_VIDEOS_PATH, 'fake')

# Configure results path
RESULTS_PATH = 'analysis_results'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Initialize the video analyzer
analyzer = VideoAnalyzer()

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to render the upload form
@app.route('/')
def index():
    return render_template('video_index.html')

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve analysis results
@app.route('/results/<path:filename>')
def analysis_file(filename):
    return send_from_directory(RESULTS_PATH, filename)

# Serve reference videos
@app.route('/reference/real/<path:filename>')
def real_reference(filename):
    return send_from_directory(REAL_VIDEOS_PATH, filename)

@app.route('/reference/fake/<path:filename>')
def fake_reference(filename):
    return send_from_directory(FAKE_VIDEOS_PATH, filename)

# Get list of reference videos
@app.route('/reference_videos')
def get_reference_videos():
    real_refs = []
    fake_refs = []
    
    # Get real reference videos
    if os.path.exists(REAL_VIDEOS_PATH):
        for file in os.listdir(REAL_VIDEOS_PATH):
            if allowed_file(file):
                real_refs.append({
                    'name': file,
                    'path': f'/reference/real/{file}'
                })
    
    # Get fake reference videos
    if os.path.exists(FAKE_VIDEOS_PATH):
        for file in os.listdir(FAKE_VIDEOS_PATH):
            if allowed_file(file):
                fake_refs.append({
                    'name': file,
                    'path': f'/reference/fake/{file}'
                })
    
    return jsonify({
        'real': real_refs,
        'fake': fake_refs
    })

# Process and analyze video
@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Generate a session ID
            session_id = str(uuid.uuid4())[:8]
            
            # Run the analysis
            results = analyzer.analyze_video(file_path, session_id=session_id)
            
            if "error" in results:
                return jsonify({'error': results["error"]}), 500
            
            # Check if this is an exact match to a reference video
            is_reference_match = results.get('exact_reference_match', False)
            
            # Get relative path to max_prob_frame for URL
            max_prob_frame = results['max_prob_frame']
            # Extract the part of the path after the session directory
            max_prob_relative = max_prob_frame
            if RESULTS_PATH in max_prob_frame:
                max_prob_relative = max_prob_frame.split(RESULTS_PATH + os.path.sep)[1]
            elif session_id in max_prob_frame:
                # Try to extract based on session ID
                parts = max_prob_frame.split(session_id + os.path.sep)
                if len(parts) > 1:
                    max_prob_relative = session_id + os.path.sep + parts[1]
            
            # Prepare the response data
            response_data = {
                'session_id': session_id,
                'video_path': os.path.basename(file_path),
                'video_url': f'/uploads/{os.path.basename(file_path)}',
                'frames_analyzed': results['frames_analyzed'],
                'fake_frames_count': results['fake_frames_count'],
                'fake_frames_percentage': results['fake_frames_percentage'],
                'average_fake_probability': results['average_fake_probability'],
                'max_fake_probability': results['max_fake_probability'],
                'overall_prediction': results['overall_prediction'],
                'max_prob_frame': os.path.basename(max_prob_frame),
                'max_prob_frame_url': f'/results/{max_prob_relative}'
            }
            
            # If this is an exact reference match, include that information
            if is_reference_match:
                match_type = results.get('match_type', '')
                response_data['exact_reference_match'] = True
                response_data['match_type'] = match_type
                
                # Create a clear message about the match
                if match_type == 'real':
                    response_data['reference_message'] = "Reference Match Found! This video exactly matches a REAL reference video in our database."
                    # Add specific real reference properties
                    response_data['reference_category'] = 'real'
                else:
                    response_data['reference_message'] = "Reference Match Found! This video exactly matches a FAKE reference video in our database."
                    # Add specific fake reference properties
                    response_data['reference_category'] = 'fake'
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    else:
        return jsonify({'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

# Add a video as a reference
@app.route('/add_reference', methods=['POST'])
def add_reference():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    category = request.form.get('category', '').lower()

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if category not in ['real', 'fake']:
        return jsonify({'error': 'Invalid category. Must be "real" or "fake"'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Add as reference video
            ref_path = analyzer.add_reference_video(file_path, category)
            
            return jsonify({
                'success': True,
                'message': f'Added {category.upper()} reference video: {os.path.basename(ref_path)}',
                'reference': {
                    'name': os.path.basename(ref_path),
                    'category': category,
                    'path': f'/reference/{category}/{os.path.basename(ref_path)}'
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'Failed to add reference video: {str(e)}'}), 500
    else:
        return jsonify({'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001) 
import os
import cv2
import torch
import numpy as np
import time
import threading
import winsound  # Windows alert sound
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
from flask import Flask, render_template_string,render_template,Response, request, redirect, url_for, session, flash ,jsonify


app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30,  n_init=1)  # Max age for tracking lost objects

# Detection Settings
restricted_area = None  # (x, y, w, h) for restricted area
crossed_ids = {}  # Store crossing events
cooldown_time = 10  # Time before detecting the same ID again
gender_labels = {}  # Store gender predictions per ID 
processing = {}  # Track IDs being processed
stop_video_flag = threading.Event()
counting_enabled = False  # Toggle person counting
detected_persons = []  # Define globally if used across functions
selected_gender = "both"



def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def play_alert():
    """Plays an alert sound when a person crosses the boundary."""
    winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500ms





def analyze_frame(person_id, face_crop):
    """Run DeepFace asynchronously for gender detection."""
    global gender_labels, processing
    processing[person_id] = True  # Mark as being processed

    try:
        result = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False)
        gender_dict = result[0]['gender']
        gender = max(gender_dict, key=gender_dict.get)  # Get highest probability
        gender_labels[person_id] = gender
        print(f"🚀 Before Checking: gender_labels = {gender_labels}")
    except Exception as e:
        print(f"DeepFace error: {e}")
        gender_labels[person_id] = "Unknown"
        


    processing[person_id] = False  # Allow next detection






def count_persons(frame):
    """Counts persons detected in the frame."""
    results = model(frame)
    count = sum(1 for box in results[0].boxes if int(box.cls[0].item()) == 0)
    return count


def boxes_intersect(box1, box2):
    """Check if two bounding boxes intersect."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)









# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
def detect_motion(frame):
    """Detect motion in the frame using background subtraction."""
    fg_mask = bg_subtractor.apply(frame)
    thresh = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)[1]  
    motion_detected = np.count_nonzero(thresh) > 500  # Adjust threshold as needed
    return motion_detected

person_counter = 0  # Global counter for unique IDs
person_tracks = {}  # Dictionary to store tracked persons

def track_person(bbox):
    """Assign a unique ID to a detected person based on bounding box."""
    global person_counter, person_tracks

    # Check if this person (bbox) is already tracked
    for pid, prev_bbox in person_tracks.items():
        if boxes_intersect(bbox, prev_bbox):  # Check overlap
            person_tracks[pid] = bbox  # Update position
            return pid  # Return existing ID

    # New person detected
    person_counter += 1
    person_tracks[person_counter] = bbox  # Store new person
    return person_counter






def generate_frames(video_source):
    """Generate video frames for streaming with improved processing."""
    global counting_enabled, restricted_area, stop_video_flag, selected_gender
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video source.")
        return

    frame_skip = 1  # Process every second frame
    frame_count = 0
    resize_factor = 0.5  # Reduce resolution to speed up processing

    while not stop_video_flag.is_set():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frame to speed up processing

        person_count = count_persons(frame) if counting_enabled else 0

        # Resize frame for detection
        small_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor),
                                         int(frame.shape[0] * resize_factor)))
        

        # Check for motion
        motion_detected = detect_motion(small_frame)
        
        if motion_detected:
            
            results = model(small_frame)  # Run YOLO only if motion is detected
            frame_alert = False
            detected_persons.clear()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()

                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = int(x1 / resize_factor), int(y1 / resize_factor), \
                                    int(x2 / resize_factor), int(y2 / resize_factor)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{result.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    
                    if cls == 0:  # Person class
                        person_id = track_person((x1, y1, x2, y2))  # Assign a unique ID
                        detected_persons.append((person_id, x1, y1, x2, y2))

                        if restricted_area and boxes_intersect((x1, y1, x2, y2), restricted_area):
                            # **Gender-Based Filtering Before Triggering Alert**
                            for person_id, x1, y1, x2, y2 in detected_persons:                   
                              print(detected_persons ,person_id , gender_labels  )
                              if person_id in gender_labels:
                                  detected_gender = gender_labels[person_id].lower()  # Convert to lowercase for consistency
                                  print(f"Detected gender for person {person_id}: {detected_gender}")
                                  print("Current gender_labels:", gender_labels)
                                                                                                   
                                  # Handling unknown gender detection
                                  if detected_gender == "unknown":
                                      frame_alert = True
                                      print("Unknown gender detected!")
                                      
                                  # Check for gender mismatch (if a specific gender is selected)
                                  elif selected_gender != "both" and detected_gender.lower() != selected_gender.lower():
                                      frame_alert = True
                                      print("Gender Mismatch Alert:", selected_gender, detected_gender)

                                  # If an alert is triggered, play sound and display alert text
                                  if frame_alert:
                                      play_alert()  # Function to play alert sound
                                      print("Playing alert sound...")
                                      cv2.putText(frame, "ALERT!", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX,
                                                  1, (0, 0, 255), 2, cv2.LINE_AA)
                                                                    
                                      
                        if person_id not in processing or not processing[person_id]:
                          face_crop = frame[y1:y2, x1:x2]

                          if face_crop.size == 0:
                              print(f"⚠️ Empty face crop for person {person_id}! Skipping analysis.")
                              continue

                          threading.Thread(target=analyze_frame, args=(person_id, face_crop)).start()

                
            # Display standby message when no motion is detected
        else:
            cv2.putText(frame, "standby mode (no motion)", 
            (10, frame.shape[0] - 80),  # Position at bottom-left
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,  # Small font
            (0, 255, 0), 1, cv2.LINE_AA)  # Green color, thin text


             #  ROI
           # Draw Transparent Restricted Area
        if restricted_area is not None:
            x1, y1, x2, y2 = restricted_area
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red filled rectangle
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # Transparency effect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Solid red border
        
        # Display gender labels
        for person_id, (pid, x1, y1, x2, y2) in enumerate(detected_persons):
            gender_text = gender_labels.get(pid, "Unknown")
            cv2.putText(frame, f"ID {pid}: {gender_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    

        cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    2, cv2.LINE_AA)

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

    cap.release()
    
@app.route('/alert_status')
def alert_status():
    return "Alert Status"

#GENDER UPDATE FROM FRONT END
@app.route('/update_gender', methods=['GET'])
def update_gender():
    global selected_gender
    selected_gender = request.args.get("gender", "both")  # Default to both
    return jsonify({"status": "updated", "selected_gender": selected_gender})   

@app.route('/clear_detection_settings', methods=['POST'])
def clear_detection_settings():
    global restricted_area, alert_status
    # Reset detection settings
    restricted_area = None
    alert_status = False
    return "OK", 200

@app.route('/')
def index():
    """Render the front page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Serve video feed."""
    stop_video_flag.clear()
    video_path = app.config.get('CURRENT_VIDEO', 'demo_browser/demo1.mp4')
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_feed')
def camera_feed():
    global stop_video_flag
    stop_video_flag = threading.Event()
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cctv_feed')
def cctv_feed():
    global stop_video_flag
    stop_video_flag = threading.Event()
    return Response(generate_frames(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_counting', methods=['POST'])
def toggle_counting():
    """Toggle person counting functionality."""
    global counting_enabled
    counting_enabled = not counting_enabled
    return redirect(url_for('index'))


@app.route('/terminate_video_feed', methods=['POST'])
def terminate_video_feed():
    global stop_video_flag
    stop_video_flag.set()
    return redirect(url_for('index'))


@app.route('/set_restricted_area', methods=['POST'])
def set_restricted_area():
    """Set restricted area coordinates."""
    global restricted_area
    x = int(request.form.get("x", 0))
    y = int(request.form.get("y", 0))
    w = int(request.form.get("w", 0))
    h = int(request.form.get("h", 0))
    restricted_area = (x, y, x + w, y + h)  # Convert to bounding box format
    
    return '', 204




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global restricted_area, alert_status
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'uploaded_video.mp4'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.config['CURRENT_VIDEO'] = file_path
            # Clear any previous detection settings when a new video is uploaded.
            restricted_area = None
            alert_status = False
            flash('Video uploaded successfully!')
            return redirect(url_for('index'))
    else:
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width,initial-scale=1.0">
          <title>Upload a Video - AI Enhanced Home</title>
          <style>
            /* Base Styles */
            body {
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
              background: linear-gradient(135deg, #141e30, #243b55);
              margin: 0; padding: 0;
              color: #e2e8f0;
            }
            h1 {
              margin: 0.5em 0; font-weight: 700;
            }
            .container {
              max-width: 600px; margin: 60px auto; padding: 30px;
              background: rgba(255,255,255,0.05);
              border-radius: 8px;
              box-shadow: 0 8px 16px rgba(0,0,0,0.6);
              backdrop-filter: blur(5px);
            }
            .container h1 {
              color: #f7fafc;
              text-shadow: 0 2px 4px rgba(0,0,0,0.6);
              text-align: center;
              margin-bottom: 20px;
            }
            /* Form Styles */
            form {
              display: flex; flex-direction: column; align-items: center;
            }
            .form-group {
              width: 100%; margin-bottom: 20px;
            }
            .form-group label {
              display: block; font-size: 1rem; margin-bottom: 8px;
              color: #f7fafc; font-weight: 600;
            }
            .form-group input[type="file"] {
              width: 100%; padding: 10px;
              border: 1px solid #4a5568;
              border-radius: 4px;
              font-size: 1rem;
              background: #2d3748;
              color: #f7fafc; outline: none;
            }
            .form-group input[type="file"]::file-selector-button {
              background-color: #2f80ed;
              color: #fff; border: none;
              padding: 8px 16px; border-radius: 4px;
              cursor: pointer;
              transition: background-color 0.3s ease, transform 0.2s;
            }
            .form-group input[type="file"]::file-selector-button:hover {
              background-color: #1366d6; transform: scale(1.02);
            }
            button {
              background-color: #e53e3e;
              color: #fff; border: none;
              padding: 12px 25px; border-radius: 4px;
              cursor: pointer; font-size: 1rem;
              transition: background-color 0.3s ease, transform 0.2s;
              text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
            }
            button:hover {
              background-color: #c53030; transform: scale(1.02);
            }
            /* Extra text style */
            .info-text {
              font-size: 0.9rem; color: #cbd5e0;
              margin-top: -10px; margin-bottom: 20px;
              text-align: center;
            }
            .info-text span {
              color: #fc8181;
            }
            /* Back Link */
            .back-link {
              display: inline-block; margin-top: 30px;
              font-size: 0.9rem; text-decoration: none;
              color: #63b3ed; transition: color 0.3s ease;
            }
            .back-link:hover {
              color: #3182ce;
            }
          </style>
        </head>
        <body>
          <div class="container">
            <h1>UPLOAD YOUR VIDEO</h1>
            <p class="info-text">
              Ready to test your luck? Upload a <span>MP4</span> file and watch our detection in action.<br>
              <em>Note:</em> Only <span>MP4</span> format is allowed.
            </p>
            <form action="/upload" method="POST" enctype="multipart/form-data">
              <div class="form-group">
                <label for="file">Select Video File</label>
                <input type="file" name="file" id="file" required>
              </div>
              <button type="submit">Upload Video</button>
            </form>
            <a class="back-link" href="{{ url_for('index') }}">← Return to the Danger Zone</a>
          </div>
        </body>
        </html>
        ''')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)
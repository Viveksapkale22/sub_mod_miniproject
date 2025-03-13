import os
import cv2
import threading
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

model = YOLO('yolov8n.pt')

stop_video_flag = threading.Event()
count_persons = False  # Flag to control person counting



def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def count_persons_in_frame(frame):
    """Count persons in the given frame using YOLOv8."""
    results = model(frame)
    person_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls[0].item()
            if int(cls) == 0:  # Class 0 is 'person'
                person_count += 1
    return person_count


def generate_frames(video_path):
    """Generate video frames for streaming."""
    global count_persons
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    while not stop_video_flag.is_set():
        success, frame = cap.read()
        if not success:
            break

        person_count = 0
        if count_persons:
            person_count = count_persons_in_frame(frame)

        # Resize and process frame
        resize_factor = 0.5
        small_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))

        results = model(small_frame)  # Run YOLO on the resized frame

        # Scale back to original frame size and draw detections manually
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cls = int(box.cls[0].item())  # Class index
                conf = box.conf[0].item()  # Confidence score

                # Scale bounding box coordinates back to original frame size
                x1, y1, x2, y2 = int(x1 / resize_factor), int(y1 / resize_factor), int(x2 / resize_factor), int(y2 / resize_factor)

                # Draw smaller bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness 2

                # Draw smaller label
                label = f"{result.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw person count on the frame
        cv2.putText(frame, f"Persons: {person_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert frame to JPEG format
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Failed to encode frame.")
            continue
        frame_bytes = buffer.tobytes()

        # Yield frame as byte data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Render the front page for login and signup."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Serve video feed for processing."""
    stop_video_flag.clear()
    video_path = app.config.get('CURRENT_VIDEO', 'demo_browser/demo1.mp4')
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_counting', methods=['POST'])
def toggle_counting():
    """Toggle person counting functionality."""
    global count_persons
    count_persons = not count_persons

    # if count_persons > 4 :
    #   flash("alert crowd detected")

    return redirect(url_for('main'))

@app.route('/terminate', methods=['POST'])
def terminate_video_feed():
    """Terminate video feed."""
    stop_video_flag.set()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle video upload."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = 'uploaded_video.mp4'  # You can implement unique naming here
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.config['CURRENT_VIDEO'] = file_path
            flash('Video uploaded successfully!')
            return redirect(url_for('index'))

    return '''
        <!doctype html>
        <title>Upload a Video</title>
        <h1>Upload a Video File</h1>
        <form action="/upload" method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
    '''



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)

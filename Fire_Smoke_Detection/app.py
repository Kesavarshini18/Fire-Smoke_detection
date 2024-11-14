from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLOv8 model
model = YOLO("D:/YOLOv8-Fire-and-Smoke-Detection-main/runs/detect/train/weights/best.pt")

video_source = None
cap = None

def generate_frames():
    global cap
    while True:
        if cap is None:
            continue
        
        success, frame = cap.read()
        if not success:
            break

        # Perform detection on the current frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Render the upload form
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_source, cap

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Set the video source to the uploaded file
        video_source = filepath
        cap = cv2.VideoCapture(video_source)

        return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)

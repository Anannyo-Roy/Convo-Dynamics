from flask import Flask , redirect , url_for , render_template ,request , Response
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import cv2
import numpy as np
from keras.models import model_from_json
from flask import Response
import pyaudio
import wave
from threading import Thread
import main

global capture,rec_frame, rec, out , timestarted

face=0
rec=0

try:
    os.mkdir('./shots')
except OSError as error:
    pass

app = Flask(__name__, static_folder='static')

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/recorded')
def recorded():
    return render_template('Live_Meeting.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/pre-recorded')
def pre_recorded():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('About_us.html')

@app.route('/success/<int:score>')
def success(score):
    return f"it have succeeded by {score}"

@app.route('/fail/<int:score>')
def fail(score):
    return f"i have failed by {score}"

@app.route('/result/<int:marks>')
def result(marks):
    if marks >=30:
        return redirect(url_for("success" ,score = marks))
    return redirect(url_for("fail" , score = marks))

@app.route('/submit3' , methods=['POST' , 'GET'])
def submit3():
    return render_template('pre-recorded.html')

@app.route('/submit' , methods=['POST' , 'GET'])
def submit():
    return render_template('recording.html')

@app.route('/submit2' , methods=['POST' , 'GET'])
def submit2():
    return render_template('live.html')

@app.route('/back' , methods = ['POST' , 'GET'])
def back():
    return render_template('index.html')


audio_stream = None
audio_frames = []
rec = False

def start_recording_audio():
    global audio_stream, audio_frames, rec , timestarted  
    audio = pyaudio.PyAudio()
    audio_stream = audio.open(format=pyaudio.paInt16,
                               channels=2,
                               rate=44100,
                               input=True,
                               frames_per_buffer=1024)
    audio_frames = []

    print("Recording audio...")

    while rec:
        data = audio_stream.read(1024)
        audio_frames.append(data)

    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()

    print("Finished recording audio.")
    # now = datetime.datetime.now()
    print(f"in audio , {timestarted}")
    wf = wave.open('static/audios/audio' + timestarted + '.wav', 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    

UPLOAD_FOLDER = "static/videos"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    actualfilename = ""
    if 'videoFile' not in request.files:
        return redirect(request.url)
    
    file = request.files['videoFile']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = file.filename
        base_filename, file_extension = os.path.splitext(filename)
        upload_folder = app.config['UPLOAD_FOLDER']
        
        # Check if the file already exists in the upload folder
        while os.path.exists(os.path.join(upload_folder, filename)):
            # If the file exists, append '(1)' to the filename before the extension
            base_filename += '(1)'
            filename = base_filename + file_extension

        # Save the file with the unique filename
        file.save(os.path.join(upload_folder, filename))
        actualfilename = filename
        print("File saved as:", filename)  # Print the filename in the terminal
    actualfilename = "static/videos/"+actualfilename
    data , overall = main.analyze_video(actualfilename)
    

    # Dictionary mapping emotion names to image file names
    emotion_image_map = {
        "Angry": "Angry.png",
        "Disgusted": "Disgusted.png",
        "Fearful": "Fearful.png",
        "Happy": "Happy.png",
        "Negative": "Negative.png",
        "Positive": "Happy.png",
        "Sad": "Sad.png",
        "Surprised": "Surprised.png",
        "Neutral": "Neutral.png"  # Adding Neutral.png as a fallback image
    }
    emotion = overall[0]
    emotion2 = overall[1]
    # Get the corresponding image file name from the dictionary
    image_file_name = emotion_image_map.get(emotion, "Neutral.png")
    image_file_name2 = emotion_image_map.get(emotion2, "Neutral.png")
    overall.append("/static/image/" + image_file_name)
    overall.append("/static/image/" + image_file_name2)
    # Append image URLs to each row in the data list
    for row in data:
        emotion = row[1]
        emotion2 = row[2]
        # Get the corresponding image file name from the dictionary
        image_file_name = emotion_image_map.get(emotion, "Neutral.png")
        image_file_name2 = emotion_image_map.get(emotion2, "Neutral.png")
        # Construct the image URL
        row.append("/static/image/" + image_file_name)
        row.append("/static/image/" + image_file_name2)
    
    return render_template('upload_form.html', data=data , overall = overall)


@app.route('/save_filename', methods=['POST'])
def save_filename():
    data = request.json
    filename = data.get('filename')
    print("Filename received:", filename)

    return 'Filename received successfully'


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Set frame rate to 60fps
    camera.set(cv2.CAP_PROP_FPS, 60)
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            break
    camera.release()

 

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def gen_frames2():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read() 
        if not success:
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

import time
video_save_dir = r'static\videos'

@app.route('/recordingrequest', methods=['POST', 'GET'])
def recc():
    global rec, out, audio_stream, audio_frames , timestarted
    rec = not rec
    audio_thread = Thread(target=start_recording_audio)
    if request.method == 'POST':
        if rec:
            now = datetime.datetime.now()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestarted = str(now).replace(":", '')
            video_file_path = os.path.join(video_save_dir, 'vid_{}.avi'.format(timestarted))
            out = cv2.VideoWriter(video_file_path, fourcc, 20.0, (1920, 1080))

            # Start new thread for recording the video
            video_thread = Thread(target=record, args=[out, ])
            video_thread.start()
            audio_thread.start()
            print("Started----------")
        elif not rec:
            out.release()
            # Stop recording audio
            # global rec
            rec = False
            time.sleep(0.5)
            print("stoped-------------------------" , timestarted )
            actualvideofile = "static/videos/"+'vid_'+timestarted+".avi"
            actualaudiofile = "static/audios/"+'audio'+timestarted+".wav"
            data , overall = main.analyze_video(actualvideofile , actualaudiofile)

            # Dictionary mapping emotion names to image file names
            emotion_image_map = {
                "Angry": "Angry.png",
                "Disgusted": "Disgusted.png",
                "Fearful": "Fearful.png",
                "Happy": "Happy.png",
                "Negative": "Negative.png",
                "Positive": "Happy.png",
                "Sad": "Sad.png",
                "Surprised": "Surprised.png",
                "Neutral": "Neutral.png"  # Adding Neutral.png as a fallback image
            }
            emotion = overall[0]
            emotion2 = overall[1]
            # Get the corresponding image file name from the dictionary
            image_file_name = emotion_image_map.get(emotion, "Neutral.png")
            image_file_name2 = emotion_image_map.get(emotion2, "Neutral.png")
            overall.append("/static/image/" + image_file_name)
            overall.append("/static/image/" + image_file_name2)
            # Append image URLs to each row in the data list
            for row in data:
                emotion = row[1]
                emotion2 = row[2]
                # Get the corresponding image file name from the dictionary
                image_file_name = emotion_image_map.get(emotion, "Neutral.png")
                image_file_name2 = emotion_image_map.get(emotion2, "Neutral.png")
                # Construct the image URL
                row.append("/static/image/" + image_file_name)
                row.append("/static/image/" + image_file_name2)

            return render_template('upload_form.html', data=data , overall = overall)
    return render_template('Live_Meeting.html')


if __name__ == '__main__':
    app.run()

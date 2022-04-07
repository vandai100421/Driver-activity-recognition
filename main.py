import argparse
from flask import Flask, flash, render_template, Response,request,json,redirect,url_for
import sqlite3
from werkzeug.utils import secure_filename
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from driver_prediction import predict_result
import numpy as np
import face_recognition
import os  # thu vien quan ly file
from datetime import datetime  # thu vien  thoi gian
# from app import app


# import urllib.request


UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
@app.route("/login",methods = ['POST', 'GET'])
def login():
    msg=""
    if request.method=='POST':
        username=request.form['username']
        with sqlite3.connect("database.db") as conn:
            try:                    
                cur=conn.cursor()
                cur.execute("INSERT INTO user (username)  values ('{0}')".format(username))
                conn.commit()                               
            except:
                conn.rollback()
                # msg="Username is existed!"    
                # return render_template('login.html',msg=msg)             
            # finally:
            #     conn.close()  
        return redirect(url_for('living_room',username=username))   
    return render_template('login.html',msg=msg)


# @app.route('/home_page')
# def home_page():
#     return render_template('index.html')
@app.route('/living_room')
def living_room():
    """Video streaming home page."""
    return render_template('living_room.html')


def gen_normal():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # time.sleep(0.1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1000)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_normal(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# nhan dien khuon mat

# lay path callable(ArithmeticError)ua file
# path = "Driver Behavior Management Software Web./static/image"
path = "./static/image"

images = []
classNames = []
myList = os.listdir(path)  # danh sach

# duyet file r them tung anh trong file vao mang images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # lay ra ten cua thu muc .jpg
# print(classNames)


# ham
def findEncodings(images):
    encodeList = []
    for img in images:
        # convert mau tu BGR thành RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  # encode
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
# print("encodeing Complete")


@app.route('/face_recog_room')
def face_recog_room():
    """Video streaming home page."""
    return render_template('face_recog_room.html')


def gen_face_recog():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        success, img = cap.read()

        # flip de lon dao vi tri trai phai cua anh
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(
            imgS)  # khung mat hien tai
        encodesCurFrame = face_recognition.face_encodings(
            imgS, facesCurFrame)  # encode khung mat hien tai

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 + 40), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1000)


@app.route('/face_recog_video')
def face_recog_video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_face_recog(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Nhan dien ngu gat

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.2
img_check = 20
detect = dlib.get_frontal_face_detector()
# Dat file is the crux of the code
predict = dlib.shape_predictor(
    "model/self_trained/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


@app.route('/drowsiness_room')
def drowsiness_room():
    """Video streaming home page."""
    return render_template('drowsiness_room.html')


def gen_drowseness():
    cap = cv2.VideoCapture(0)

    flag = 0
    while True:
        ret, img = cap.read()
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= img_check:
                    cv2.putText(img, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #print ("Drowsy")
            else:
                flag = 0

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/drowsiness_video')
def drowsiness_video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_drowseness(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





# Nhan dien hanh vi

@app.route('/behavior_room')
def behavior_room():
    """Video streaming home page."""
    return render_template('behavior_room.html')

# INPUT_VIDEO_FILE = "input_video.mp4"

# INPUT_VIDEO_FILE = "/home/vandai1042001/Desktop/Distracted-Driver-Detection-master/Save video/05-04-2022 - 19:48:12.mp4"

now = datetime.now()
now = now.strftime("%d-%m-%Y-%H:%M:%S")

OUTPUT_VIDEO_FILE = "Save video/" + str(now) + ".mp4"

def gen_behavior():
    
    vs = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction
        # frame = cv2.flip(frame, 1)
        output = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        # frame -= mean
        frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5

        # make predictions on the frame and then update the predictions
        # queue
        label = predict_result(frame)
        # preds = model.predict(np.expand_dims(frame, axis=0))[0]
        # Q.append(preds)
        
        # perform prediction averaging over the current history of
        # previous predictions
        # results = np.array(Q).mean(axis=0)
        # i = np.argmax(results)
        # label = lb.classes_[i]

            # draw the activity on the output frame
        text = "activity: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.25, (0, 255, 0), 5)
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30,
                (W, H), True)
        # write the output frame to disk
        writer.write(output)
        # show the output image
        # cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        frame = cv2.imencode('.jpg', output)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
    cv2.destroyAllWindows()
    vs.release()


@app.route('/behavior_video')
def behavior_video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_behavior(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# nhan dien hanh vi trong lich su

@app.route('/lichsu_room')
def lichsu_room():
	return render_template('lichsu_room.html')

@app.route('/', methods=['GET' ,'POST'])
def upload_video():
    
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		return render_template('lichsu_room.html', filename=filename)

# @app.route('/display/<filename>')
# def display_video(filename):
# 	#print('display_video filename: ' + filename)
# 	return redirect(url_for('static', filename='uploads/' + filename), code=301)
# filepath = '/home/vandai1042001/Desktop/main/static/Save video/05-04-2022-19:49:05.mp4'
filepath = 'input_video.mp4'
def gen_behavior_ls(filename):
    tmp = str(filename)
    input = filepath + tmp
    print(input)
    vs = cv2.VideoCapture(filepath)

    writer = None
    (W, H) = (None, None)
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction
        # frame = cv2.flip(frame, 1)
        output = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        # frame -= mean
        frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5

        # make predictions on the frame and then update the predictions
        # queue
        label = predict_result(frame)
        # preds = model.predict(np.expand_dims(frame, axis=0))[0]
        # Q.append(preds)
        
        # perform prediction averaging over the current history of
        # previous predictions
        # results = np.array(Q).mean(axis=0)
        # i = np.argmax(results)
        # label = lb.classes_[i]

            # draw the activity on the output frame
        text = "activity: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.25, (0, 255, 0), 5)
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30,
                (W, H), True)
        # write the output frame to disk
        writer.write(output)
        # show the output image
        # cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        frame = cv2.imencode('.jpg', output)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
    cv2.destroyAllWindows()
    vs.release()




@app.route('/display/<filename>')
def display_video(filename):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_behavior_ls(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')







##################################
if __name__ == ("__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int,
                        default=5000, help="Running port")
    parser.add_argument("-H", "--host", type=str,
                        default='0.0.0.0', help="Address to broadcast")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)

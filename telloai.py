import sys
import traceback
import tellopy
import av
import cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import os
import face_recognition
import datetime
import imutils
import numpy as np
import argparse
from imutils.face_utils import FaceAligner
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
achyut_image = face_recognition.load_image_file("achyut.JPG")
achyut_face_encoding = face_recognition.face_encodings(achyut_image)[0]

# Load a second sample picture and learn how to recognize it.



# Create arrays of known face encodings and their names
known_face_encodings = [
    achyut_face_encoding
]
known_face_names = [
    "Achyut"

]
21
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='./resnet10/deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='./resnet10/res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save", action='store_true',
                help="save the video")
args = vars(ap.parse_args())


def handleFileReceived(event, sender, data):
    global date_fmt
    # Create a file in ~/Pictures/ to receive image data from the drone.
    path = '%s/tello-%s.jpeg' % (
        os.getenv('HOMEPATH'),  # Changed from Home to Homepath
        datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    with open(path, 'wb') as fd:
        fd.write(data)
    # print('Saved photo to ',path)


if args["save"]:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (400, 300))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


def main():
    global startY, startX, endX, endY
    drone = tellopy.Tello()
    landed = True
    speed = 30
    up, down, left, right, forw, back, clock, ctclock = False, False, False, False, False, False, False, False
    ai = True
    pic360 = False
    currentPic = 0
    move360 = False
    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        container = av.open(drone.get_video_stream())
        drone.subscribe(drone.EVENT_FILE_RECEIVED, handleFileReceived)
        # skip first 300 frames
        frame_skip = 300
        while True:
            try:
                for frame in container.decode(video=0):
                    if process_this_frame:
                        # Resize frame of video to 1/4 size for faster face recognition processing
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                        rgb_small_frame = small_frame[:, :, ::-1]

                        # Find all the faces and face encodings in the current frame of video
                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                        face_names = []
                        for face_encoding in face_encodings:
                            # See if the face is a match for the known face(s)
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"

                            # # If a match was found in known_face_encodings, just use the first one.
                            # if True in matches:
                            #     first_match_index = matches.index(True)
                            #     name = known_face_names[first_match_index]

                            # Or instead, use the known face with the smallest distance to the new face
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                            face_names.append(name)

                    process_this_frame = not process_this_frame

                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    if 0 < frame_skip:
                        frame_skip = frame_skip - 1
                        continue
                    start_time = time.time()
                    image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                    image = imutils.resize(image, width=400)
                    (h, w) = image.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                                 (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()

                    face_dict = {}

                    for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with the
                        # prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the `confidence` is
                        # greater than the minimum confidence
                        if confidence < 0.5 and name=="Achyut":
                            continue

                        # compute the (x, y)-coordinates of the bounding box for the
                        # object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{:.2f}%".format(confidence * 100)
                        face_dict[text] = box

                    # Will go to face with the highest confidence
                    try:
                        H, W, _ = image.shape
                        distTolerance = 0.05 * np.linalg.norm(np.array((0, 0)) - np.array((w, h)))

                        box = face_dict[sorted(face_dict.keys())[0]]
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(image, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)

                        distance = np.linalg.norm(np.array((startX, startY)) - np.array((endX, endY)))

                        if int((startX + endX) / 2) < W / 2 - distTolerance:
                            # print('CounterClock')
                            drone.counter_clockwise(30)
                            ctclock = True
                        elif int((startX + endX) / 2) > W / 2 + distTolerance:
                            # print('Clock')
                            drone.clockwise(30)
                            clock = True
                        else:
                            if ctclock:
                                drone.counter_clockwise(0)
                                ctclock = False
                                # print('CTClock 0')
                            if clock:
                                drone.clockwise(0)
                                clock = False
                                # print('Clock 0')

                        if int((startY + endY) / 2) < H / 2 - distTolerance:
                            drone.up(30)
                            # print('Up')
                            up = True
                        elif int((startY + endY) / 2) > H / 2 + distTolerance:
                            drone.down(30)
                            # print('Down')
                            down = True
                        else:
                            if up:
                                up = False
                                # print('Up 0')
                                drone.up(0)

                            if down:
                                down = False
                                # print('Down 0')
                                drone.down(0)

                        # print(int(distance))

                        if int(distance) < 110 - distTolerance:
                            forw = True
                            # print('Forward')
                            drone.forward(30)
                        elif int(distance) > 110 + distTolerance:
                            drone.backward(30)
                            # print('Backward')
                            back = True
                        else:
                            if back:
                                back = False
                                # print('Backward 0')
                                drone.backward(0)
                            if forw:
                                forw = False
                                # print('Forward 0')
                                drone.forward(0)


                    except Exception as e:
                        # print(e)
                        None

                    if args["save"]:
                        out.write(image)

                    cv2.imshow('Original', image)

                    # cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                    if frame.time_base < 1.0 / 60:
                        time_base = 1.0 / 60
                    else:
                        time_base = frame.time_base
                    frame_skip = int((time.time() - start_time) / time_base)
                    keycode = cv2.waitKey(1)

                    if keycode == 32:
                        if landed:
                            drone.takeoff()
                            landed = False
                        else:
                            drone.land()
                            landed = True

                    if keycode == 27:
                        raise Exception('Quit')

                    if keycode == 13:
                        drone.take_picture()
                        time.sleep(0.25)
                        # pic360 = True
                        # move360 = True

                    if keycode & 0xFF == ord('q'):
                        pic360 = False
                        move360 = False

            except Exception as e:
                print(e)
                break


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        if args["save"]:
            out.release()
        drone.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

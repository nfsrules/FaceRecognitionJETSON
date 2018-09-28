from functionsbis import *
import sys
import argparse
import cv2
from keras.models import model_from_json, load_model
import face_recognition
import time
import tensorflow as tf



def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    args = parser.parse_args()
    return args



def read_cam(cap, model, gender_model):
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN
    process_this_frame = True
    while True:
      
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, frame = cap.read() # grab the next image frame from camera
        if show_help:
            cv2.putText(frame, help_text, (11, 20), font,
                        1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(frame, help_text, (10, 20), font,
                        1.0, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #rgb_small_frame = small_frame[:, :, ::-1]  # size (180, 320, 3)
        
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame,number_of_times_to_upsample=1, model='cnn')
            face_emotions = []
            for face in face_locations:
                #with tf.device("/gpu:0"):
                # Get face coordinates
                top, right, bottom, left = face

                # Crop image 
                face_image = small_frame[top:bottom, left:right]
                
                # Pre-process image
                gray_img = pre_processing(face_image)
                
                # Call model.predict() method
                previous_time=time.time()
                net_output = model.predict(gray_img)
                net2_output = gender_model.predict(gray_img)
                current_time=time.time()
                print(current_time-previous_time)
                # Decode output
                prediction = decode(net_output)
                gender_prediction = gender_decoder(net2_output)
                
                # Add emotion to rectangle name
                face_emotions.append(prediction + '\n' + gender_prediction)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), emotion in zip(face_locations, face_emotions):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom + 50), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        # Display the resulting image
        cv2.imshow(WINDOW_NAME, frame)
        #cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(2)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


def main():
    WINDOW_NAME = 'CameraDemo'
    gender_model = load_model('Models/trained_GenderNet.hdf5')
    # load json and create model
    json_file = open('Models/face_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("Models/face_model.h5")
    print("Loaded model from disk")
    model = loaded_model

    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = open_cam_onboard()

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)

    read_cam(cap, model, gender_model)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
   # with tf.Session() as sess:
    main()


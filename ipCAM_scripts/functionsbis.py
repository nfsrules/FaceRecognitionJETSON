import sys
import numpy as np
import cv2

WINDOW_NAME = 'CameraDemo'

# Functions 
def decode(net_output):
    '''EmotionNet output decoder.
    
    '''
    if (np.argmax(net_output) == 0):
        prediction = 'Angry'
    elif (np.argmax(net_output) == 1):
        prediction = 'Disgust'
    elif (np.argmax(net_output) == 2):
        prediction = 'Fear'
    elif (np.argmax(net_output) == 3):
        prediction = 'Happy'
    elif (np.argmax(net_output) == 4):
        prediction = 'Sad'
    elif (np.argmax(net_output) == 5):
        prediction = 'Surprise'
    else:
        prediction = 'Neutral'
        
    return prediction
        
    
def pre_processing(color_image_rectangle):
    '''Transform input image to EmotionNet input size.
    
    '''
    # Convert image to B&W
    gray_img = cv2.cvtColor(color_image_rectangle,cv2.COLOR_BGR2GRAY)

    # Resize image
    gray_img = cv2.resize(gray_img,(48,48), interpolation = cv2.INTER_CUBIC)

    # Reshape and add mini-batch dimension
    gray_img = np.reshape(gray_img,(48,48,1))
    gray_img = np.expand_dims(gray_img, axis=0)
                      
    return gray_img

def gender_encoder(gender):
    '''Gender label encoder --------------------------
    
    Male: 0
    Female: 1
    '''
    if gender == 'm':
        gender = 0
    else:
        gender = 1
    return gender


def gender_decoder(net_output, threshold=0.65):
    '''GenderNet label decoder -----------------------

    0: male
    1: female
    threshold: Probability minimum threshold to assign a label.
    '''
    if np.max(net_output) > threshold:
        if np.argmax(net_output) == 0:
            prediction = 'male'
            probability = np.max(net_output)
        else:
            prediction = 'female'
            probability = np.max(net_output)
    else:
        prediction = ''
        probability = ''

    return prediction
                    
# Tegra functions
def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)RGB ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard():
    return cv2.VideoCapture('M6.mp4')


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')


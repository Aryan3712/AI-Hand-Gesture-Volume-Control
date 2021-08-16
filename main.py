import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

# pycaw imports
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# parameters
w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
previous_time = 0

# creating an object of HandTracking Module
# also setting detection_confidence as 0.7 so that the hand movement on the screen
# does not flicker and the machine makes sure that what it is detecting is really a hand
detector = htm.HandDetector(detection_confidence=0.7)

# pycaw initializations
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_bar = 400
vol = 0
volume_per = 0
# print(volume.GetVolumeRange())
# volume range is (-65.25, 0.0, 0.03125)

volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

while True:
    success, img = cap.read()
    img = detector.find_Hands(img)
    lm_list = detector.find_positions(img, draw=False)
    # in the below code we need the value of landmark 4 and 8
    # as for controlling the volume we need the tip of our thumb and the
    # the tip of our index finger respectively
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])
        # storing the values
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        # finding the center of the line
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # creating a circle around the tip
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        # creating a line between the tips
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # creating a circle in the middle of the line
        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        # we require the length of this line to measure and control volume
        # finding length of the line using math
        line_length = math.hypot(x2-x1, y2-y1)
        # print(line_length)

        # since the minimum hand range detected is 50 (300:maximum)
        # Hand range 50 - 300
        # Volume range = -65 - 0
        # our task is to convert the hand range into volume range
        # which is achieved by interp() of numpy

        vol = np.interp(line_length, [50, 300], [min_vol, max_vol])
        volume_bar = np.interp(line_length, [50, 300], [400, 150])
        volume_per = np.interp(line_length, [50, 300], [0,100])

        # print(int(line_length), vol)

        # setting the vol finally as the master volume
        volume.SetMasterVolumeLevel(vol, None)

        if line_length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # creating a volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
    cv2.rectangle(img, (50, int(volume_bar)), (85, 400), (255, 0, 0), cv2.FILLED)

    # fps
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, "Volume: " + str(int(volume_per))+" %", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Volume Control Gesture", img)
    cv2.waitKey(1)

import cv2
import mediapipe as mp
import time


class HandDetector():

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)
        # the Hands class only uses RBG images
        self.mp_draw = mp.solutions.drawing_utils

    def find_Hands(self, img, draw=True):
        # converting the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # processing the frame and giving the result
        results = self.hands.process(imgRGB)
        # to see the results (detecting hands)
        # print(results.multi_hand_landmarks)
        # checking if result is not null
        # and drawing landmarks for each hand (maximum 2 hands are allowed)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_positions(self, img, hand_number=0, draw=True):
        lm_list = []
        imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # processing the frame and giving the result
        results = self.hands.process(imRGB)
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[hand_number]
            # below lm gives the ratio of the image as x and y
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                # we will find out the height and the width of the image
                h, w, c = img.shape
                # finding the positions
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    previous_time = 0
    current_time = 0
    # video object (webcam 0)
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        # passing the img to the find_hands of HandDetector
        img = detector.find_Hands(img)
        lm_list = detector.find_positions(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        # finding and printing the fps
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, "FPS: " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("HandTracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

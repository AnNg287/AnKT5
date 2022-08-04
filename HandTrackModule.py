import cv2
from matplotlib.pyplot import draw             
import mediapipe as mp  
import time
import os
import HandTrackCounter as htm

cap = cv2.VideoCapture(0)
folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    overlayList.append(image)
pTime = 0
detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      maxHands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while True:
   
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    img= detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    h, w, c = overlayList[0].shape
    img[0:h, 0:w] = overlayList[0]
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    if len(lmList) != 0:
            fingers = []

            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            print(totalFingers)

            h, w, c = overlayList[totalFingers-1].shape
            img[0:h, 0:w] = overlayList[totalFingers-1]
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 15)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (450,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
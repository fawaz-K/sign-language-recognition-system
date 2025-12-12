import os
import cv2
import time
import json
import numpy as np
from Hand_detector import hand_detector



cap = cv2.VideoCapture(1)
detector = hand_detector(maxHands=1, minTrackCon=0.8, detectionCon=0.8)


imgSize = 112
counter = 0
SaveImages = False
Data = []
imgs = []
vectors = []

time.sleep(1)
label = input('Enter the label: ')
File = "Data/hand_data.json"

os.makedirs(f'Data/{label}', exist_ok=True)

if not os.path.exists(File):
    with open(File, 'w') as f:
        json.dump([], f)

with open(File, 'r') as f:
    try:
        Data = json.load(f)
    except json.JSONDecodeError:
        Data = []


while True:
    try: 
        img = cv2.flip(cap.read()[1], 1)
        h, w = img.shape[:2]
        canvas = np.ones((imgSize, 2*imgSize, 3), np.uint8)

        img, hands = detector.detectHands(img, True)

        if hands:
            hand = hands[0]            
            imglm, rLmList = detector.get_img_lm(hand, imgSize, (255,0,255), 3, (255,255,255), 2)
            imgHand = detector.get_hand_shape(img, hand, imgSize)

            canvas[:, : imgSize] = imglm
            canvas[:, imgSize: 2*imgSize] = imgHand

            vector = (np.array(rLmList)/imgSize).flatten().tolist()

            if SaveImages:
                directory = f'Data/{label}/IMG_{int(time.time()*1000)}.jpg'

                cv2.imwrite(directory, canvas)
                vectors.append(vector)
                imgs.append(directory)

                if counter%30 == 0 and counter!=0:
                    Data.append({
                        'imgs': imgs,
                        'vectors': vectors,
                        'label': label
                    })

                    imgs = []
                    vectors = []

                counter+=1
                print(counter)


        cv2.imshow('Frame', img)
        cv2.imshow('Canvas', canvas)
        
        key = cv2.waitKey(1)

        if key == 27: break

        elif key == ord('s') and hands:
            SaveImages = not SaveImages

        elif key == ord("n"):
            cv2.destroyWindow('Frame')
            cv2.destroyWindow('Canvas')

            label = input("Enter the label: ")

            os.makedirs(f'Data/{label}', exist_ok=True)

            counter = 0
            SaveImages = False


    except cv2.error:
        pass


with open(File, 'w') as f:
    json.dump(Data, f, indent=4)

cv2.destroyAllWindows()
cap.release()
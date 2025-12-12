import cv2
import json
import numpy as np
import tensorflow as tf
from Hand_detector import hand_detector



model_path = "Data/model.h5"
imgSize = 112
counter = 0 

with open("Data/labels.json", "r") as f:
    labels = json.load(f)
labels = {f'{v}': k for k, v in labels.items()}


model = tf.keras.models.load_model(model_path)
detector = hand_detector(maxHands=1, minTrackCon=0.8, detectionCon=0.8)
cap = cv2.VideoCapture(1)

vectors = []
imgs = []

predicted_label = None

while True:
    img = cv2.flip(cap.read()[1], 1)
    canvas = np.ones((imgSize, 2*imgSize, 3), np.uint8)
    
    img, hands = detector.detectHands(img, True)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imglm, rLmList =  detector.get_img_lm(hand, imgSize, (255,0,255), 3, (255,255,255), 2)

        vector = (np.array(rLmList)/imgSize).flatten().tolist()
        
        imgHand = detector.get_hand_shape(img, hand, imgSize)
        canvas[:, :imgSize] = imglm
        canvas[:, imgSize:] = imgHand

        vectors.append(vector)
        imgs.append(canvas.astype(np.float32) / 255.0)
        counter+=1

        if counter == 30:
            vectors = np.expand_dims(np.array(vectors), axis=0)
            imgs = np.expand_dims(np.array(imgs), axis=0)  
            predictions = model.predict([vectors, imgs], verbose=0)
            predicted_index = np.argmax(predictions)
            predicted_label = labels[f'{predicted_index}']

            counter = 0
            vectors = []
            imgs = []

        
        cv2.putText(img, f'{predicted_label}', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.imshow('ASL Prediction', img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
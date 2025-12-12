import cv2
import math
import numpy as np
import mediapipe as mp
from typing import overload, Tuple





class hand_detector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.offset = 10


    def detectHands(self, img: np.ndarray, flipped: bool):
        h, w = img.shape[:2]
        hands = []

        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            for handType, hand_lm in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                hand: dict = {}
                lmList = []
                xList = []
                yList = []

                for lm in hand_lm.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    lmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                bbox = min(xList) - self.offset, min(yList) - self.offset, max(xList) - min(xList) + 2*self.offset, max(yList) - min(yList) + 2*self.offset

                if not flipped:
                    if handType.classification[0].label == "Right":
                        Htype = "Left"
                    else:
                        Htype = "Right"
                else:
                    Htype = handType.classification[0].label

                hand['lmList'] = lmList
                hand['bbox'] = bbox
                hand['type'] = Htype
                hands.append(hand)


                bbox = bbox[0] - 10, bbox[1] - 10, bbox[2] + 15, bbox[3] + 15
                cv2.rectangle(img, bbox, (255,0,255), 2)
                cv2.line(img, (bbox[0], bbox[1]), (bbox[0] + 30, bbox[1]), (255,255,255), 9)
                cv2.line(img, (bbox[0], bbox[1]), (bbox[0], bbox[1] + 30), (255,255,255), 9)
                
                cv2.line(img, (bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2] - 30, bbox[1]), (255,255,255), 9)
                cv2.line(img, (bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1] + 30), (255,255,255), 9)
                
                cv2.line(img, (bbox[0],bbox[1] + bbox[3]), (bbox[0] + 30,bbox[1] + bbox[3]), (255,255,255), 9)
                cv2.line(img, (bbox[0],bbox[1] + bbox[3]), (bbox[0],bbox[1] + bbox[3] - 30), (255,255,255), 9)
                
                cv2.line(img, (bbox[0] + bbox[2],bbox[1] + bbox[3]), (bbox[0] + bbox[2] - 30,bbox[1] + bbox[3]), (255,255,255), 9)
                cv2.line(img, (bbox[0] + bbox[2],bbox[1] + bbox[3]), (bbox[0] + bbox[2],bbox[1] + bbox[3] - 30), (255,255,255), 9)

        return img, hands


    def draw_landmarks(self, img: np.ndarray, lmList, circle_color, circle_r, line_color, line_t):
        for i, lm in enumerate(lmList):
            if i != 4 and i != 8 and i != 12 and i != 16 and i != 20:
                cv2.line(img, lm, lmList[i + 1], line_color, line_t)

        palm_lines = [(1,5), (5,9), (9,13), (13,17), (0,17)]
        for start, end in palm_lines:
            cv2.line(img, lmList[start], lmList[end], line_color, line_t)

        for lm in lmList:
            cv2.circle(img, lm, circle_r, circle_color, cv2.FILLED)



    def get_cropped_img(self, img: np.ndarray, bbox: list, cropped_img_size: int):
        x, y, w, h = bbox
        imgCrop = img[y: y + h, x: x + w].copy()
        imgBg = np.ones((cropped_img_size, cropped_img_size, 3), np.uint8)


        if h > w:
            new_w = math.floor((cropped_img_size * w) / h)
            imgCrop = cv2.resize(imgCrop, (new_w, cropped_img_size))
            center = (cropped_img_size - new_w) // 2
            imgBg[:, center: center + new_w] = imgCrop

        else:
            new_h = math.floor((cropped_img_size * h) / w)
            imgCrop = cv2.resize(imgCrop, (cropped_img_size, new_h))
            center = (cropped_img_size - new_h) // 2
            imgBg[center: center + new_h, :] = imgCrop

        return imgBg
    


    def get_hand_shape(self, img, hand, img_hand_size):
        lmList = hand['lmList']
        bbox = hand['bbox']


        h, w = img.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(hand_mask, [np.array(lmList, dtype=np.int32)], 255)

        kernel = np.ones((5, 5), np.uint8)
        hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
        hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)

        hand_on_black_background = np.where(hand_mask_blurred[..., None] > 0, img, np.zeros_like(img))

        
        grayscale_frame = cv2.cvtColor(hand_on_black_background, cv2.COLOR_BGR2GRAY)
        expanded_frame = np.stack((grayscale_frame,) * 3, axis=-1)

        return self.get_cropped_img(expanded_frame, bbox, img_hand_size)



    @overload
    def get_img_lm(
        self,
        hand: dict,
        imgSize: int,
        circle_color: tuple,
        circle_r: int,
        line_color: tuple,
        line_t: int,
    ) -> Tuple[np.ndarray, list]: pass
    


    @overload
    def get_img_lm(
        self,
        img: np.ndarray,
        imgSize: int,
        flipped: bool,
        circle_color: tuple,
        circle_r: int,
        line_color: tuple,
        line_t: int,
    ) -> Tuple[list, np.ndarray, list]: pass
    


    def get_img_lm(self, *args, **kwargs):
        if isinstance(args[0], dict):
            hand = args[0]
            imgSize, circle_color, circle_r, line_color, line_t = args[1:]
            img_lm = np.zeros((imgSize, imgSize, 3), np.uint8)
            rLandmarks = []

            lmList = hand['lmList']
            bbox = hand['bbox']

            if bbox[3] > bbox[2]:
                new_w = math.floor((imgSize * bbox[2]) / bbox[3])
                offset_x = (imgSize - new_w) // 2
                offset_y = 0
                scale = imgSize / bbox[3]
            else:
                new_h = math.floor((imgSize * bbox[3]) / bbox[2])
                offset_x = 0
                offset_y = (imgSize - new_h) // 2
                scale = imgSize / bbox[2]

            for i in range(len(lmList)):
                rx = int((lmList[i][0] - bbox[0]) * scale) + offset_x
                ry = int((lmList[i][1] - bbox[1]) * scale) + offset_y
                rLandmarks.append([rx, ry])

            self.draw_landmarks(img_lm, rLandmarks, circle_color, circle_r, line_color, line_t)
            return img_lm, rLandmarks

        else: 
            img = args[0]
            imgSize, flipped, circle_color, circle_r, line_color, line_t = args[1:]
            img, hands = self.detectHands(img, flipped)

            if hands:
                img_lm, lm = self.get_img_lm(hands[0], imgSize, circle_color, circle_r, line_color, line_t)
                return hands, img_lm, lm

            return [], np.ones((imgSize, imgSize, 3), np.uint8), []
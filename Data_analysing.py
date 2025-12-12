import json
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import (Input, Dense, Flatten, Conv2D, MaxPooling2D,
                                     Dropout, LSTM, TimeDistributed, concatenate)
import os


SEQ_LEN = 30
imgH = 112
imgW = imgH*2
imgC = 3
file = "Data/hand_data.json"   


with open(file, "r") as f:
    data = json.load(f) 

X_vectors, X_images, y_labels = [], [], []
label_to_index, current_index = {}, 0

for item in data:
    
    vectors = np.array(item["vectors"], dtype=np.float32)
    if len(vectors) != SEQ_LEN:
        continue  
    X_vectors.append(vectors)

    
    frames = []
    for frame_path in item["imgs"]:
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (imgW, imgH))
        img = img.astype(np.float32) / 255.0
        frames.append(img)
    frames = np.array(frames)
    if frames.shape[0] != SEQ_LEN:
        continue
    X_images.append(frames)

    
    label = item["label"]
    if label not in label_to_index:
        label_to_index[label] = current_index
        current_index += 1
    y_labels.append(label_to_index[label])

X_vectors = np.array(X_vectors)   
X_images = np.array(X_images)     
y_labels = np.array(y_labels)     

num_classes = len(label_to_index)

print("Dataset loaded:")
print("X_vectors:", X_vectors.shape)
print("X_images:", X_images.shape)
print("y_labels:", y_labels.shape)



vector_input = Input(shape=(SEQ_LEN, X_vectors.shape[2]))
x1 = LSTM(128, return_sequences=False)(vector_input)
x1 = Dense(64, activation="relu")(x1)


frame_input = Input(shape=(SEQ_LEN, imgH, imgW, imgC))

cnn = tf.keras.Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),
    Flatten()
])

x2 = TimeDistributed(cnn)(frame_input)  
x2 = LSTM(128, return_sequences=False)(x2)  
x2 = Dense(128, activation="relu")(x2)


merged = concatenate([x1, x2])
merged = Dense(128, activation="relu")(merged)
merged = Dropout(0.3)(merged)
output = Dense(num_classes, activation="softmax")(merged)

model = Model(inputs=[vector_input, frame_input], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()


model.fit([X_vectors, X_images], y_labels,
          batch_size=8, epochs=20, validation_split=0.2)


os.makedirs("Data", exist_ok=True)
model.save("Data/model.h5")
with open("Data/labels.json", "w") as f:
    json.dump(label_to_index, f, indent=4)

print("Model + labels saved.")
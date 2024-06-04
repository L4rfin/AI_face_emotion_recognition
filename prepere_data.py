import numpy as np
import shutil
from utils import get_face_landmarks
import os
import cv2

data_dir = "./dataset/train"
output = []
i = 0
for emotion_index, emotion in enumerate(os.listdir(data_dir)):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)
        image = cv2.imread(image_path)
        face_landmarks = get_face_landmarks(image)
        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_index))
            output.append(face_landmarks)
            print(face_landmarks)
            i += 1
        if i > 15:
            i = 0
            break
        shutil.move(image_path, 'dataset/bin/'+emotion+'/'+image_path_)
with open("data.txt", 'ab') as f:
    np.savetxt(f, output)

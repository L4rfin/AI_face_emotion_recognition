import pickle
import cv2

from utils import get_face_landmarks

# emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emotions = ['angry', 'happy', 'neutral', 'sad', 'surprised']

with open('./model', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, static_image_mode=False)
    output = model.predict([face_landmarks])
    print(output)
    cv2.putText(frame,
                emotions[int(output[0])],
                (10, frame.shape[0] - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

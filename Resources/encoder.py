import cv2
import face_recognition
import pickle
import os

path = 'KnownFaces'
images = []
studentIds = []
for filename in os.listdir(path):
    img = cv2.imread(f'{path}/{filename}')
    images.append(img)
    studentIds.append(os.path.splitext(filename)[0])

encodeListKnown = []
for img in images:
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    if encodings:
        encodeListKnown.append(encodings[0])

encodeListKnownWithIds = [encodeListKnown, studentIds]
with open('EncodeFile.p', 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)

print("Encoding Complete and File Saved")

# from ultralytics import YOLO
import cv2
import os
import cvzone
import math
import pickle
import face_recognition

cap = cv2.VideoCapture(0) # For Webcam
cap.set(3, 640)
cap.set(4, 480)
print(cap.get(3), cap.get(4))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4") # From Video

imgBackground = cv2.imread("./Resources/background.png")

# importing the mode images
folderModePath = "Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    # imgModeList.append(cv2.imread(folderModePath + "/" + path))
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
# print(modePathList)
# print(imgModeList)
# print(len(imgModeList))

#Load the encoding file
print("Loading Encoded File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encoded File Loaded")

namesDict = {"0": "No Match", "20200626_202614": "Emily"}

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)



    #
    imgBackground[182:182+480, 455:455+640] = img
    # imgBackground[44:44+633, 808:808+414] = imgModeList[0]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        x_offset = 455
        y_offset = 182

        # Get best match
        matchIndex = faceDis.argmin()
        print("matchIndex: ", matchIndex)

        top, right, bottom, left = faceLoc
        # Compute scale between imgS and img
        scale = img.shape[1] / imgS.shape[1] # typically 4.0

        # Scale back up because you resized the image to 0.25
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)

        w = right - left
        h = bottom - top

        # Map to background coordinates (because webcam is pasted at (55, 162))
        x_draw = x_offset + left
        y_draw = y_offset + top

        # Color based on match
        color = (0, 255, 0) if matches[matchIndex] else (0, 0, 255)

        name = namesDict.get(studentIds[matchIndex], "Unknown")

        if faceDis[matchIndex] > 0.6:
            name = "No Match"

        # Draw rectangle using cvzone
        cvzone.cornerRect(imgBackground, (x_draw, y_draw, w, h), rt=0, colorC=color)
        cvzone.putTextRect(imgBackground, f'{name}', (x_draw, y_draw), scale=1.5, thickness=2, colorT=(255, 255, 255), colorR=color )

        for match in matches:
            if match:
                print("Match Found")

        print("matches: ", matches)
        print("faceDis: ", faceDis)


    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)


# results = model("Images/sb.jpg", show=True)
# annotated = results[0].plot() # get the image with boxes drawn
# cv2.imshow("Result", annotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
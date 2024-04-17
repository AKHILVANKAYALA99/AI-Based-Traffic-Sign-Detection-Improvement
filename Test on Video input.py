import random
import cv2
import numpy as np
from ultralytics import YOLO

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def remove_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def increase_brightness(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# opening the file in read mode
my_file = open(r"V:\Hackathons\Techgium\final.txt")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO(r"C:\Users\RAMPALLI VAMSI\Downloads\best (410).pt")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("inference/videos/afriq0.MP4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    #frame=remove_noise(frame)
    frame=remove_blur(frame, kernel_size=(5, 5))

    frame=increase_brightness(frame, factor=1.5)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    
    detect_params = model.predict(source=[frame], conf=0.5, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Traffic Sign Detection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
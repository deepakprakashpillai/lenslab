from flask import Flask, render_template
import cv2
import torch
import numpy as np
from PIL import Image

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

app = Flask(__name__)


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/home')
def home():
    return render_template('homepage.html')


@app.route('/product')
def product():
    return render_template('latestindex.html')


@app.route('/payment')
def payment():
    return render_template('payment.html')


@app.route('/measure')
def measure():

    MARGIN_OF_ERROR = 0.88

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        if auto:  # minimum rectangle
            dw = np.mod(new_shape[1] - new_unpad[0], 32) / 2  # width padding
            dh = np.mod(new_shape[0] - new_unpad[1], 32) / 2  # height padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]
        else:  # pad
            dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
            dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img

    def scale_coords(img_shape, coords, img0_shape):
        gain = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])
        pad_x = (img_shape[1] - img0_shape[1] * gain) / 2.0
        pad_y = (img_shape[0] - img0_shape[0] * gain) / 2.0

        if isinstance(coords, np.ndarray):
            coords = coords.copy()
        elif isinstance(coords, list):
            coords = np.array(coords)

        if len(coords.shape) > 1:
            coords[:, 0] -= pad_x
            coords[:, 2] -= pad_x
            coords[:, 1] -= pad_y
            coords[:, 3] -= pad_y
            coords[:, :4] /= gain
            coords[:, :4] = np.clip(coords[:, :4], 0, img0_shape[1])

        return coords

    def detect_faces(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def plot_one_box(x, img, color=(0, 255, 0), line_thickness=3):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        w = int(x[2]) - int(x[0])
        return w

    def detect_credit_card(image):
        # Preprocess the image
        img0 = letterbox(image, new_shape=(640, 640))
        w = 1
    # Convert image to RGB and normalize
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

    # Inference
        device = select_device('')

    # Run the image through the model
        pred = model(img.to(device))[0]

    # Apply non-maximum suppression to get the most confident detections
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

    # Process the detections
        for det in pred:
            if det is not None and len(det):
                # Iterate over the detections and draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    # label = f'{model.names[int(cls)]} {conf:.2f}'
                    xyxy = scale_coords(
                        img.shape[2:], xyxy, img0.shape).round()

                    cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(
                        xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    w = int(xyxy[2]) - int(xyxy[0])

        return [image, w]


# Specify the path to your .pt model file
    model_path = './best.pt'

# Load the model
    model = attempt_load(model_path)

# Set up the webcam
    cap = cv2.VideoCapture(0)

    while True:
        iterator = 0
        avg = 0
        wid = 0
    # Read the frame from the webcam
        ret, frame = cap.read()
        face_width = 1
        card_width = 2
        card_results = detect_credit_card(frame)

        card_boxes = card_results[0]
        card_width = card_results[1]

# Scale the card bounding boxes
        card_xyxy = []
        for box in card_boxes:
            if len(box) == 4:  # Check if the box format is (x, y, w, h)
                x, y, w, h = box
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                card_xyxy.append([x1, y1, x2, y2])
            elif len(box) == 2:  # Check if the box format is (x1, y1, x2, y2)
                x1, y1, x2, y2 = box
                card_xyxy.append([x1, y1, x2, y2])
            else:
                # Handle other box formats if necessary
                pass


# Perform face detection
        faces = detect_faces(frame)

# Draw the face bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_width = w

# Draw the card bounding boxes
        for box in card_xyxy:
            card_width = plot_one_box(box, frame)


# Calculate the ratio
        ratio = face_width / card_width
        actual_face_width = ratio * 85.6
        actual_face_width = actual_face_width * MARGIN_OF_ERROR
        if iterator == 0:
            avg = actual_face_width
            wid = 0
        else:
            avg = (actual_face_width + avg)/2
            if iterator % 10 == 0:
                wid = avg

        iterator = iterator + 1

        if actual_face_width > 100 and actual_face_width < 120:
            req_size = "Extra Narrow"
        elif actual_face_width < 130:
            req_size = "Narrow"
        elif actual_face_width < 140:
            req_size = "Medium"
        elif actual_face_width < 150:
            req_size = "Wide"
        elif actual_face_width < 160:
            req_size = "Extra-wide"
# Draw the ratio on the frame
        if actual_face_width > 100 and actual_face_width < 160:
            text = req_size
        else:
            text = "Not recognised"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Combined Detections', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()

    return render_template('measure_index.html')


@app.route('/men')
def men_page():
    return render_template('Men.html')


@app.route('/women')
def women_page():
    return render_template("women.html")


@app.route('/kids')
def kids_page():
    return render_template('kids.html')


if __name__ == "__main__":
    app.run(debug=True)

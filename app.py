import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
import sys
import requests
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
import json
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_pumpkin = load_lottiefile("hmm.json")

def object_detector(image_data):
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Error: The image was not loaded properly. Please check the file and try again.")
        return

    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    height, width, channels = img.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), font, 2, (127, 255, 0), 2)

    st.image(img, caption='Processed Image.', use_column_width=True)

mtcnn = MTCNN()

# Define a function to check if a face is clearly visible
def no_of_face(image_data, confidence_threshold=0.8):
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        st.error("Error: The image was not loaded properly. Please check the file and try again.")
        return

    faces = mtcnn.detect_faces(image)
    count_high_confidence_faces = 0

    for face in faces:
        if face['confidence'] > confidence_threshold:
            count_high_confidence_faces += 1

    return count_high_confidence_faces

def Live_face():
    class FaceDetectionApp(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("Face Detection App")
            self.setGeometry(100, 100, 800, 600)

            # Create a QLabel to display the webcam feed
            self.video_display = QLabel(self)
            self.video_display.setAlignment(Qt.AlignCenter)

            # Create a QPushButton to start and stop face detection
            self.start_stop_button = QPushButton("Start Detection", self)

            # Create a layout to organize widgets
            layout = QVBoxLayout()
            layout.addWidget(self.video_display)
            layout.addWidget(self.start_stop_button)

            # Create a central widget to hold the layout
            central_widget = QWidget()
            central_widget.setLayout(layout)

            self.setCentralWidget(central_widget)

            # Initialize OpenCV VideoCapture
            self.cap = cv2.VideoCapture(0)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)

            # Connect the button click event
            self.start_stop_button.clicked.connect(self.start_stop_detection)
            self.detection_active = False

        def start_stop_detection(self):
            if not self.detection_active:
                self.start_stop_button.setText("Stop Detection")
                self.detection_active = True
                self.timer.start(30)
            else:
                self.start_stop_button.setText("Start Detection")
                self.detection_active = False
                self.timer.stop()

        def update_frame(self):
            ret, frame = self.cap.read()
            if ret:
                # Perform face detection using a pre-trained classifier (e.g., Haar Cascade)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Convert the frame to a format suitable for displaying in a PyQt window
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)

                # Display the frame in the QLabel
                self.video_display.setPixmap(pixmap)

        def closeEvent(self, event):
            self.cap.release()

    if __name__ == '__main__':
        app = QApplication(sys.argv)
        window = FaceDetectionApp()
        window.show()
        sys.exit(app.exec_())

st.set_page_config(page_title="Project on ML", page_icon="🧊", layout="wide", initial_sidebar_state="auto")
# Page Title
st.title("Project on ML")
st.title(" ")

# Sidebar
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Select a Page", ("Home", "Overview", "Object Detector", "Face Detector", "Made By"))

# Home Page
if selected_page == "Home":
    st.header("Welcome to the Machine Learning Project")
    st.header(" ")
    st.write("In today's fast-paced world, the utilization of cutting-edge technology for various applications has become more critical than ever. Machine learning, a subset of artificial intelligence, has revolutionized the way we approach tasks such as image and video processing, object detection, and facial recognition. These advancements have paved the way for innovative solutions in various domains, from security and healthcare to entertainment and beyond.")
    st.write("This project embarks on a journey into the realm of machine learning, focusing on the integration of powerful libraries and techniques to address real-world challenges. Our primary tools in this endeavor include OpenCV (Open Source Computer Vision Library) and MTCNN (Multi-Task Cascaded Convolutional Networks). OpenCV provides us with a robust and comprehensive framework for computer vision tasks, while MTCNN serves as a state-of-the-art model for efficient and accurate facial detection and recognition.")
    # Display an image
    st.image("123.jpg", caption="Machine Learning", use_column_width=True)
    st.header("Project Goals")
    st.write("1. Exploring OpenCV: OpenCV is an open-source computer vision and machine learning software library that plays a pivotal role in this project. We aim to leverage its capabilities to manipulate and process images, detect objects, and perform a wide array of computer vision tasks.")
    st.write("2. Facial Detection with MTCNN: Facial recognition is a critical aspect of our project. By utilizing MTCNN, we seek to achieve accurate and reliable facial detection, enabling us to analyze facial features and expressions with precision.")
    st.write("3. Integration of PyQt for GUI: To provide an intuitive and user-friendly interface, we will incorporate PyQt, a set of Python bindings for the Qt application framework. This allows us to create graphical user interfaces (GUIs) that facilitate easy interaction with our machine learning models.")
    st.write("4. Practical Applications: Throughout the project, we will explore practical applications of machine learning in various fields, such as surveillance, access control, and identity verification. These applications highlight the real-world significance of our work.")
    st.write("5. Future Prospects: As we delve into the capabilities of machine learning and computer vision, we will also discuss potential future developments and enhancements, showcasing how these technologies are poised to shape the future.")
    st.header("Acknowledgements")
    st.write("This project would not have been possible without the valuable contributions of the OpenCV and MTCNN communities, as well as the support of our mentors and collaborators. We extend our gratitude to all those who have played a role in advancing the field of machine learning and computer vision.")
    st_lottie(lottie_pumpkin, speed=1, width=800, height=600, key="pumpkin")
    
    # Display a video
    st.header("Project Video")
    st.video("https://youtu.be/8fFJmPS1qd8")

# Overview Page
elif selected_page == "Overview":
    st.header("Overview of the Project")
    st.write("This project involves the following key components:")
    st.write("- Object detection using YOLO")
    st.write("- Face detection using MTCNN")
    st.write("- Real-time video feed processing")
    st.write("- GUI for live face detection")
    st.image("overview.jpg", caption="Project Overview", use_column_width=True)

# Object Detector Page
elif selected_page == "Object Detector":
    st.header("Object Detection using YOLO")
    st.write("Upload an image to detect objects:")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        object_detector(uploaded_file)

# Face Detector Page
elif selected_page == "Face Detector":
    st.header("Face Detection using MTCNN")
    st.write("Upload an image to detect faces:")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        num_faces = no_of_face(uploaded_file)
        st.write(f"Number of faces detected: {num_faces}")

# Made By Page
elif selected_page == "Made By":
    st.header("Made By")
    st.write("This project is developed by:")
    st.write("- Developer 1")
    st.write("- Developer 2")
    st.write("We have utilized OpenCV, MTCNN, and PyQt for this project.")
    st.image("team.jpg", caption="Our Team", use_column_width=True)

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


def object_detector(xyz):
    file_bytes = np.asarray(bytearray(xyz.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Error: The image was not loaded properly. Please check the path and try again.")
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
    scale = 0.5
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    # st.write("Height:", height, "Width:", width, "Channels:", channels)
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


    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("Height:", height, "Width:", width, "Channels:", channels)

mtcnn = MTCNN()

# Define a function to check if a face is clearly visible
def no_of_face(image_path,h=0.8):
    # Load the image
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        st.error("Error: The image was not loaded properly. Please check the file and try again.")
        return
    # Detect faces in the image
    faces = mtcnn.detect_faces(image)

    count_high_confidence_faces = 0

    if not faces:
        return 0
    
    for face in faces:
        if face['confidence'] > h:
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
    st.write("1. Exploring OpenCV: OpenCV is an open-source computer vision and machine learning software library that plays a pivotal role in this project. We aim to delve into its vast capabilities for image and video analysis, object tracking, and feature extraction. Our project will showcase the flexibility of OpenCV in performing tasks such as face detection, image manipulation, and more.")
    st.write("2. MTCNN for Facial Recognition: The Multi-Task Cascaded Convolutional Networks (MTCNN) model is renowned for its ability to perform real-time facial detection with high accuracy. In this project, we leverage MTCNN to create a facial recognition system capable of detecting faces, extracting facial features, and recognizing individuals. This is particularly valuable in applications related to security, authentication, and personalized services.")
    st.write("3. User Interface with Streamlit: To make our machine learning applications accessible and user-friendly, we employ Streamlit, a Python library that enables us to develop interactive web applications. We will design user interfaces that allow users to interact with our machine learning models, providing an intuitive experience.")
    st.write("4. Real-World Applications: Throughout the project, we will focus on real-world applications of our machine learning solutions. Whether it's enhancing security through facial recognition, automating image processing, or developing interactive applications, we aim to demonstrate the practicality and impact of machine learning in various domains.")
    st.write(" ")
    st.write("The world of machine learning is dynamic, with endless possibilities waiting to be explored. This project serves as a testament to the potential of machine learning when combined with powerful libraries like OpenCV and MTCNN. Our journey begins here, where we embrace the challenges, the innovation, and the excitement of shaping the future with machine learning.")
    # Show a button
    if st.button("Learn More"):
        st.write("You can learn more about machine learning on https://www.ibm.com/topics/machine-learning.")

# Overview
if selected_page == "Overview":
    st.header("Overview")
    st.title(" ")
    st.write("Broadly, this project can be divided into two parts those are _Object detection and _Face Detection, further _Face Detection can also be divided into two sub parts those are face detection for saved image and for real-time photage(Live).")
    st.title(" ")
    st.header("Object Detection") 
    st.write("Object detection is a fundamental computer vision task with countless practical applications. This project leverages state-of-the-art deep learning techniques, specifically YOLO, in conjunction with the versatile OpenCV library and the comprehensive COCO dataset, to achieve accurate and efficient object detection.")   
    st.image("object.jpeg", caption="Object Detection", use_column_width=True)
    st.write("Key Components: ")
    st.write("1. YOLO-based Object Detection: This project utilizes the YOLO (You Only Look Once) model, known for its speed and accuracy in real-time object detection. YOLO divides images into a grid and makes predictions at each grid cell to identify multiple objects simultaneously.")
    st.write("2. Integration with OpenCV: OpenCV (Open Source Computer Vision Library) is the backbone of this project, providing a wide array of tools for image and video processing, including object detection. OpenCV is used to read, process, and visualize images and video streams.")
    st.write("3. The COCO Dataset: The COCO dataset is a benchmark for object detection tasks, containing over 200,000 images across 80 object categories. This dataset plays a crucial role in training and evaluating the object detection model, ensuring its ability to recognize a wide variety of objects.")
    st.title(" ")
    st.header("_Face Detection")
    st.header("Real-time")
    st.write("Real-time face detection is a significant application in the field of computer vision and human-computer interaction. This code, developed using Python with PyQt5, OpenCV (cv2), and associated libraries, focuses on a user-friendly graphical interface to perform real-time face detection.")
    st.write("The code combines the powerful computer vision capabilities of OpenCV with the intuitive PyQt5 framework to create an interactive experience for us. By leveraging the Qt graphical user interface, the application offers a seamless real-time face detection solution.")
    st.image("real.jpeg", caption="Object Detection", use_column_width=True)
    st.write("Key Components: ")
    st.write("1. Interactive GUI: The code utilizes PyQt5 to build an interactive graphical user interface, providing users with a user-friendly and visually appealing environment.")
    st.write("2. Real-Time Face Detection: The heart of the application lies in the real-time face detection capability enabled by OpenCV. The code harnesses OpenCV's pre-trained deep learning models for robust and accurate face detection.")
    st.write("3. User Configurability: Users can configure settings, such as the choice of camera source and confidence threshold, to customize the face detection experience according to their specific requirements.")
    st.write("4. Visual Feedback: Detected faces are highlighted with bounding boxes and labeled with confidence scores, providing visual feedback to users about the detection process.")
    st.title(" ")
    st.header("Saved Image")
    st.write("The project presents a solution for detecting the number of faces in images using the MTCNN (Multi-Task Cascaded Convolutional Networks) model in conjunction with the OpenCV library. It extends beyond face detection by introducing the concept of face detection confidence to ensure the reliability of the results.")
    st.write("In this project, we employ the MTCNN model, a state-of-the-art deep learning model for face detection. MTCNN excels in its ability to efficiently detect faces while providing confidence scores that indicate the accuracy of each detection. We leverage this feature to count the number of faces with high confidence, typically set at a threshold of 0.9 or above. Even you can set the confidence value by yourself, according to your need.")
    st.image("savedimage2.jpeg", caption="Object Detection", use_column_width=True)
    st.write("Key Components:")
    st.write("1. MTCNN for Face Detection: The Multi-Task Cascaded Convolutional Networks (MTCNN) model is a prominent choice for accurate face detection. It employs cascaded networks to identify faces in an image efficiently. The output not only includes the bounding box coordinates but also a confidence score for each detected face.")
    st.write("2. OpenCV for Image Processing: OpenCV is a powerful computer vision library that provides extensive support for image processing and manipulation. We use OpenCV for image loading and to draw bounding boxes around detected faces.")
    st.write("3. Face Detection Confidence: In face detection, confidence scores play a crucial role in distinguishing high-confidence detections from false positives. In this project, we set a confidence threshold (e.g., 0.9) to identify faces with a high level of confidence.")
    st.title(" ")
    st.title(" ")


# Object detector
if selected_page == "Object Detector":
    st.write("Here is the real thing: ")
    # Page Title
    st.title("Object Detector")

    # user_input = st.text_input("Enter the image name for object detection :", "")

    # if st.button("Detect"):
    #     # Ask the user for a string input
        
    #     # Display the user's input
    #     object_detector(user_input)
    #     st.write(f"You entered: {user_input}")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        object_detector(uploaded_file)
        # st.write(f"You entered: {uploaded_file}")


# Face detector
if selected_page == "Face Detector":
    
    st.title("Image Face Detector")

    st.write("Note: Higher the confidence_threshold value, lower the probability of detecting blur face.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    user_input2 = st.number_input("Enter the confidence thershold :", min_value=0.0, max_value=1.0)
    if st.button("Detect"):
        # Ask the user for a string input
        
        # Display the user's input
        num_faces=no_of_face(uploaded_file,user_input2)
        st.image(uploaded_file, caption="Image you selected", use_column_width=True)
        st.write(f"In above image, total {num_faces} faces are visible which are having confidence value above {user_input2}.")

    
    st.write("Here is the real thing: ")
    
    st.title("Real-Time Face Detector")
    st.write("Note: Keep the camera clear.")

    if st.button("Start Detecting"):
        # Display the user's input
        Live_face()

    st.title(" ")
    st.title(" ")


if selected_page == "Made By":
    st.title(" ")
    st.title("Avinash")
    
    st.title(" ")
    st.write("Made as Minor Project(2024).")
    st.write("You can contact me at avinashkumarsaxena115@gmail.com.")
    st_lottie(
        lottie_pumpkin,
        speed=1,
        reverse=False,
        loop=True,
        quality="high"
    )

# Footer
st.text("© 2024 Machine Learning -by Avinash")

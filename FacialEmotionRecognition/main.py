import cv2
import gradio as gr
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(input_data):
    if isinstance(input_data, str):  # If input is a file path
        frame = cv2.imread(input_data)
    else:  # If input is from webcam
        frame = input_data

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,
                    result[0]['dominant_emotion'],
                    (x, y - 10),
                    font, 1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)

    return frame[:, :, ::-1]  # convert BGR to RGB for Gradio

iface = gr.Interface(fn=detect_emotion, inputs="image", outputs="image", title="Emotion Detection")
iface.launch(share=True,  debug=True)

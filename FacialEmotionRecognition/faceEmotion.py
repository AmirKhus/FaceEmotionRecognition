{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dcde092d-a249-49b8-afa9-85c79d6d280b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "from deepface import DeepFace\n",
    "\n",
    "faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')\n",
    "\n",
    "cap = cv2.VideoCapture(1)\n",
    "if not cap.isOpened():\n",
    "    cap = cv2.VideoCapture(0)\n",
    "if not cap.isOpened():\n",
    "    raise IOError(\"Cannot open Webcam\")\n",
    "\n",
    "while True:\n",
    "    ret, frame = cap.read()\n",
    "    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)\n",
    "\n",
    "    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "    faces = faceCascade.detectMultiScale(gray, 1.1, 4)\n",
    "\n",
    "    for (x, y, w, h) in faces:\n",
    "        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)\n",
    "\n",
    "        font = cv2.FONT_HERSHEY_SIMPLEX\n",
    "\n",
    "        cv2.putText(frame,\n",
    "            result[0]['dominant_emotion'],\n",
    "            (x, y - 10),\n",
    "            font, 1,\n",
    "            (0, 0, 255),\n",
    "            2,\n",
    "            cv2.LINE_4)\n",
    "\n",
    "\n",
    "    cv2.imshow('Original video', frame)\n",
    "\n",
    "    if cv2.waitKey(2) & 0xFF == ord('q'):\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe14933e-fdff-4fe4-b6ef-48b494d0e2e4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

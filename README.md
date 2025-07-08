 ✋ Hand Gesture Recognition using CNN and OpenCV 🎥🧠

This project detects and recognizes hand signs (0–9 & unknown) in real time using a Convolutional Neural Network (CNN) and OpenCV.

![demo](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.8%2B-blue) ![tensorflow](https://img.shields.io/badge/tensorflow-2.x-orange)

---

## 🔍 Project Overview

This system:
- Trains a CNN model to classify hand gestures representing numbers (0–9).
- Uses OpenCV to capture live video and detect gestures inside a Region of Interest (ROI).
- Predicts and displays the detected number in real-time.

---

## 📁 Dataset

- Source: [Kaggle - Sign Language for Numbers](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers)
- Contains black-and-white images of hands representing numbers 0–9.

---

## 🧰 Tech Stack

- `Python`
- `TensorFlow / Keras`
- `OpenCV`
- `NumPy`
- `Scikit-learn`
- `Kaggle API`

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository


git clone https://github.com/your-username/hand-gesture-recognition.git

cd hand-gesture-recognition



2️⃣ Install Dependencies
Ensure you're using a virtual environment (venv or conda).
pip install tensorflow opencv-python scikit-learn kaggle pandas


3️⃣ Configure Kaggle API
Download your kaggle.json from your Kaggle account and place it inside:
mkdir %USERPROFILE%\.kaggle

copy kaggle.json %USERPROFILE%\.kaggle\


4️⃣ Train the Model
Run the training script to build and save the model:

python train_model.py


5️⃣ Run Real-Time Prediction
Run the live webcam script:

python handgestures.py


📊 Model Architecture
Conv2D(32) → MaxPooling2D →
Conv2D(64) → MaxPooling2D →
Conv2D(128) → MaxPooling2D →
Flatten → Dense(128) → Dropout(0.5) → Dense(11, softmax)



✅ To-Do / Improvements
 Improve accuracy using data augmentation 🌀

 Add grayscale preprocessing if needed

 Use MediaPipe for better hand detection 🤖

 Deploy as a web or mobile app 🌐📱



🙌 Acknowledgements
Dataset: Kaggle Sign Language for Numbers

TensorFlow & OpenCV teams



💡 Tips
If the model isn't predicting accurately, ensure consistent hand position, lighting, and background.

Your webcam must be accessible and not used by other apps.





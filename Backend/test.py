import dlib

path = r"C:\Users\SHALINI\Desktop\python programs\New folder\datasets\shape_predictor_68_face_landmarks.dat"

try:
    predictor = dlib.shape_predictor(path)
    print("Model loaded successfully!")
except RuntimeError as e:
    print("Error loading model:", e)

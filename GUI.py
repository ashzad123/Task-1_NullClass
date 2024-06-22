import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained model
model_path = 'Model.h5'
model = load_model(model_path)

# Function to preprocess and predict age and gender
def predict_age_gender(image_path):
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (if necessary)
        image = cv2.resize(image, (128, 128))  # Resize image to model input size
        image = image / 255.0  # Normalize pixel values

        # Make prediction
        age_pred, gender_pred = model.predict(np.expand_dims(image, axis=0))

        # Convert predictions to readable format
        predicted_age = int(age_pred[0])
        predicted_gender = 'Male' if np.argmax(gender_pred[0]) == 1 else 'Female'

        return predicted_age, predicted_gender

    except Exception as e:
        messagebox.showerror("Error", f"Error predicting age and gender: {str(e)}")

# Function to handle image selection and prediction
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_age, predicted_gender = predict_age_gender(file_path)
        result_label.config(text=f"Predicted Age: {predicted_age}\nPredicted Gender: {predicted_gender}")

# Create main application window
root = tk.Tk()
root.title("Age and Gender Prediction")

# Create GUI components
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

# Run the GUI main loop
root.mainloop()

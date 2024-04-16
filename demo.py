import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained face detection model
model = load_model('face_detection_model.h5')

def detect_faces(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Use the model to predict the bounding boxes
    y_pred = model.predict(image_array)

    # Process the predicted bounding boxes
    bboxes = []
    for i in range(y_pred.shape[1]):
        if np.all(y_pred[0, i] == 0):
            break
        x1, y1, x2, y2 = y_pred[0, i]
        bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    return bboxes

def main():
    st.title("Face Detection Demo")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button to trigger face detection
        if st.button("Detect Faces"):
            # Detect faces in the image
            bboxes = detect_faces(image)

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)

            st.image(image, caption='Image with Detected Faces', use_column_width=True)

if __name__ == "__main__":
    main()
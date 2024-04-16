import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the trained face detection model
path_to_best_weights = r'C:\Users\hemas\Documents\Applied_AI_and_ML_Courses\Foundations_Of_ML\labs\CSCN8010\runs\detect\train5\weights\best.pt'
fine_tuned_model = YOLO(path_to_best_weights)

def main():
    st.title("Face Detection Demo")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)

        # Button to trigger face detection
        if st.button("Detect Faces"):
            # Detect faces in the image
            result = fine_tuned_model.predict(image)
            bbox = result[0].boxes.xywh
            if bbox.numel() == 0:
                st.markdown(r"$\Large\textbf{The image does not contain human face}$", unsafe_allow_html=True)
            else:
                res_plotted = result[0].plot()
                st.image(res_plotted, caption='Image with Detected Faces', use_column_width=True)

if __name__ == "__main__":
    main()
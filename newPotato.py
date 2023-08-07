import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Function to make predictions on the uploaded image using the ML model
def make_prediction(model, image):
    # Preprocess the image
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    # Make prediction
    predictions = model.predict(image_array)
    return predictions

def main():
    # Load the pre-trained ML model from .h5 file
    model_path = "potato_leaf_detection.h5"  # Replace with the actual path to your .h5 file
    model = load_model(model_path)

    # Set the app title
    st.title("Image Classification App")

    # Add an uploader widget
    st.write("Upload an image for classification")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make predictions if an image is uploaded
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to make predictions
        if st.button("Make Prediction"):
            with st.spinner("Making prediction..."):
                # Make prediction using the ML model
                predictions = make_prediction(model, image)

                # Display the prediction results
                st.write("Prediction Results:")
                for i, score in enumerate(predictions[0]):
                    st.write(f"Class {i}: {score:.4f}")

if __name__ == "__main__":
    main()

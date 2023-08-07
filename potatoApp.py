import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Classification of Potato Leaf using the Leaves",
    layout="wide"  # Add this line to enable the dark theme
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('potato_leaf_detection.h5')
    return model

with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Potato Disease Classification
         """
         )

def is_image(file):
    try:
        img = Image.open(file)
        img.verify()  # Check if the file is a valid image
        return True
    except:
        return False

def is_leaf_image(image_data):
    try:
        img = Image.open(image_data)
        img.verify()  # Check if the image is valid
        return np.any(np.array(img))  # Check if the image contains any pixels
    except:
        return False

file = st.file_uploader("", type=["jpg", "png", "jpeg", "heic"])

def import_and_predict(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    if is_image(file):
        if is_leaf_image(file):
            image = Image.open(file)
            st.image(image, use_column_width=True)
            predictions = import_and_predict(image, model)
            class_names = ['Early blight', 'Late blight', 'Healthy']

            st.write("Prediction Results:")
            for i, class_name in enumerate(class_names):
                probability = predictions[0][i]
                confidence = probability * 100
                st.write(f"{class_name}: {confidence:.2f}%")

            predicted_class_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_index]
            st.write(f"Prediction: {predicted_class} with confidence {predictions[0][predicted_class_index] * 100:.2f}%")
            if predicted_class == 'Healthy':
                st.success("Classified as Healthy")
            else:
                st.warning(f"Classified as {predicted_class}")
        else:
            st.warning("Not a leaf image! Please upload an image of a leaf.")
    else:
        st.warning("Not a valid image file! Please upload an image (jpg, png, jpeg, or heic).")

temp = """"""
st.markdown(temp, unsafe_allow_html=True)

import logging
import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

# Configure logging to write logs to a file
logging.basicConfig(filename="image_classification.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the saved model
def load_model():
    try:
        logging.info("Loading model...")
        model = tf.keras.models.load_model('Children_vs_Adults_Classification.h5')
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error("Error loading model: %s", str(e))

# Load the image and preprocess it
def load_image(image_file):
    try:
        logging.info("Loading image...")
        # Extract the file content from the UploadedFile object
        img_content = image_file.read()
        # Convert the file content to a TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img_content, dtype=tf.string)
        # Use tf.io.decode_image to decode the image content
        img = tf.io.decode_image(img_tensor, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        logging.info("Image loaded successfully.")
        return img
    except Exception as e:
        logging.error("Error loading image: %s", str(e))

# Classify the image
def classify_image(model, img):
    try:
        logging.info("Classifying image...")
        predictions = model.predict(np.array([img]))
        class_names = ['Child', 'Adult']  
        class_indices = np.argmax(predictions, axis=1)
        if(predictions[0] <0.5):
            class_names[class_indices[0]]= 'Adult'
            predictions[0]= 1- predictions[0]
        logging.info("Image classified successfully.")
        return class_names[class_indices[0]], predictions[0]
    except Exception as e:
        logging.error("Error classifying image: %s", str(e))

# Display the classification result
def display_result(class_name, confidence):
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"# Prediction: {class_name}")
            st.write(f"Confidence: {confidence:.2f}")

        with col2:
            if (class_name=='Adult'):
                value= [confidence, (1-confidence)]
            else:
                value= [(1-confidence), confidence]
            data = pd.DataFrame({
                'Category': ['Adult', 'Child'],
                'Confidence': value
            })
            # Define colors for each category
            category_colors = alt.Scale(domain=['Adult', 'Child'],
                            range=[ '#ff7f0e', '#2ca02c'])

            # Create a bar chart using Altair
            chart = alt.Chart(data).mark_bar(width=25).encode(
                x=alt.X('Category', scale=alt.Scale(padding=10)),
                y='Confidence',
                color=alt.Color('Category', scale=category_colors)
            )
            st.write(chart)
    except Exception as e:
        logging.error("Error displaying result: %s", str(e))

# Define the main function for the app
def main():
    try:
        st.set_page_config(page_title="Image Classification App", page_icon=":guitar:", layout="wide")
        st.title("Children vs Adult Classification")
        st.write("Upload an image to classify it!")

        # Load the model
        model = load_model()

        # Load the image and preprocess it
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        logging.info("File uploaded: %s", image_file)  # Log statement
        if image_file:
            img = load_image(image_file)

            # Classify the image
            class_name, predictions = classify_image(model, img)

            # Display the classification result
            display_result(class_name, predictions[np.argmax(predictions)])
    except Exception as e:
        logging.error("An error occurred: %s", str(e))

if __name__ == "__main__":
    main()

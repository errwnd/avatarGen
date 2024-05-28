import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64

# Set the matplotlib backend to 'Agg' to avoid threading issues
plt.switch_backend('Agg')

def load_face_generator_model(model_path):
    return load_model(model_path)

def generate_image(model, size):
    noise = tf.random.normal([1, 100])
    generated_image = model(noise)
    generated_image_resized = tf.image.resize(generated_image, [size, size])
    return generated_image_resized

def normalize_image(image):
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    return image

def plot_image(generated_image, size):
    normalized_image = normalize_image(generated_image[0]).numpy()
    fig, ax = plt.subplots(figsize=(5, 5))  # Larger figure size for better quality
    ax.imshow(normalized_image)
    ax.axis('off')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
    img_io.seek(0)
    plt.close(fig)
    return img_io

def create_download_link(img_io):
    b64 = base64.b64encode(img_io.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="generated_image.png">Download Image</a>'
    return href

def run_app(model_path='face_generator_02_50.h5'):
    model = load_face_generator_model(model_path)
    
    st.title("Image Generator")
    st.write("Generate custom images using a trained model.")
    
    size = st.slider("Select Image Size", min_value=64, max_value=512, value=128)
    
    if st.button("Generate"):
        generated_image_resized = generate_image(model, size)
        img_io = plot_image(generated_image_resized, size)
        
        # Display the image at a smaller size for better visual quality
        st.image(normalize_image(generated_image_resized[0]).numpy(), width=200, caption="Generated Image")
        
        download_link = create_download_link(img_io)
        st.markdown(download_link, unsafe_allow_html=True)

if __name__ == '__main__':
    run_app()

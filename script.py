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

def enhance_contrast(image, factor=1.5):
    mean = tf.reduce_mean(image)
    return tf.clip_by_value((image - mean) * factor + mean, 0, 1)

def plot_image(generated_image, size):
    enhanced_image = enhance_contrast(normalize_image(generated_image[0])).numpy()
    fig, ax = plt.subplots(figsize=(3.6, 3.6))  # Larger figure size for better quality
    ax.imshow(enhanced_image)
    ax.axis('off')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
    img_io.seek(0)
    plt.close(fig)
    return img_io

def create_download_link(img_io):
    b64 = base64.b64encode(img_io.getvalue()).decode()
    href = f'''
    <a href="data:file/png;base64,{b64}" download="generated_image.png">
        <button style="background-color:#4CAF50; border:none; color:white; padding:15px 32px; text-align:center; text-decoration:none; display:inline-block; font-size:16px; margin:4px 2px; cursor:pointer;">Download Image</button>
    </a>
    '''
    return href

def run_app(model_path='face_generator_REFINED_120.h5'):
    model = load_face_generator_model(model_path)
    
    st.title("Image Generator")
    st.write(" ")
    
    size = st.slider("Select Image Size", min_value=64, max_value=512, value=128)
    
    if st.button("Generate"):
        generated_image_resized = generate_image(model, size)
        img_io = plot_image(generated_image_resized, size)
        
        # Display the enhanced image at a smaller size for better visual quality
        st.image(enhance_contrast(normalize_image(generated_image_resized[0])).numpy(), width=200, caption="Generated Image")
        
        download_link = create_download_link(img_io)
        st.markdown(download_link, unsafe_allow_html=True)

if __name__ == '__main__':
    run_app()

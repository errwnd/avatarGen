import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64

# Set the matplotlib backend to 'Agg' to avoid threading issues
plt.switch_backend('Agg')

# Load model
model_path = 'face_generator_Final_50.h5'  # Update this path
model = load_model(model_path)

st.title("AvatarGen")
st.write("Generate custom avatars using a trained model.")

# Slider for adjusting the image size
size = st.slider("Select Image Size", min_value=10, max_value=100, value=36)

# Button to generate image
if st.button("Generate"):
    # Generate image
    noise = tf.random.normal([1, 100])
    generated_image = model(noise)

    # Resize the generated image to the specified size
    generated_image_resized = tf.image.resize(generated_image, [size, size])

    # Plot the generated image
    fig, ax = plt.subplots(figsize=(size / 10, size / 10))  # Adjust figure size to match the image dimensions
    ax.imshow(generated_image_resized[0, :, :, :])
    ax.axis('off')

    # Save the plot to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
    img_io.seek(0)
    plt.close(fig)

    # Display the image in Streamlit
    st.image(generated_image_resized[0, :, :, :], use_column_width=True, caption="Generated Image")

    # Provide a download link for the image
    b64 = base64.b64encode(img_io.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="generated_image.png">Download Image</a>'
    st.markdown(href, unsafe_allow_html=True)

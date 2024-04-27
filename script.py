from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st


model = load_model("faces.h5")
generator = model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#generator=model.compile(optimizer="Adam",metrics=['accuracy'],)



def main():
    st.title("avatarGen")

  if st.button("Generate"):
    noise = tf.random.normal([1, 100])
    generated_image = model(noise)
    print("Generated image shape: ", generated_image.shape)
    z=plt.imshow(generated_image[0,:,:,:])
    st.image('z')
if __name__ == "__main__":
    main()

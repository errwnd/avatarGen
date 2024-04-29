from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st


##model = load_model("faces.h5")
#generator = model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#generator=model.compile(optimizer="Adam",metrics=['accuracy'],)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2DTranspose):
        layer_config = layer.get_config()
        layer_config.pop('groups', None)
        layer.__init__(**layer_config)



def main():
    st.title("avatarGen")
if __name__ == "__main__":
    main()

import os
import speech_recognition as sr
from PIL import Image
import streamlit as st
import time

# initialize the speech recognition engine
r = sr.Recognizer()

# define function to display sign language images
def display_images(text):
    # get the file path of the images directory
    img_dir = "images/"

    # initialize variable to track image position
    image_pos = st.empty()

    # iterate through the text and display sign language images
    for char in text:
        if char.isalpha():
            # display sign language image for the alphabet
            img_path = os.path.join(img_dir, f"{char}.png")
            img = Image.open(img_path)

            # update the position of the image
            image_pos.image(img, width=150)

            # wait for 2 seconds before displaying the next image
            time.sleep(2)

            # remove the image
            image_pos.empty()
        elif char == ' ':
            # display space image for space character
            img_path = os.path.join(img_dir, "space.png")
            img = Image.open(img_path)

            # update the position of the image
            image_pos.image(img, width=150)

            # wait for 2 seconds before displaying the next image
            time.sleep(2)

            # remove the image
            image_pos.empty()

    # wait for 2 seconds before removing the last image
    time.sleep(2)
    image_pos.empty()


# add start button to start recording audio
if st.button("Start"):
    # record audio for 5 seconds
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source, phrase_time_limit=5)

        try:
            text = r.recognize_google(audio)
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand what you said.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")

    # convert text to lowercase
    text = text.lower()
    # display the final result
    st.write(f"You said: {text}")

    # display sign language images
    display_images(text)

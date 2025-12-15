import streamlit as st

st.title("Audio Input Demo")

# Display the audio input widget
audio_value = st.audio_input("Record a voice message")

# Check if audio data has been submitted
if audio_value:
    st.success("Audio recorded successfully!")
    # Play back the recorded audio
    st.audio(audio_value)
    
    # You can also save the audio to a file
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_value.read())

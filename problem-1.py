import streamlit as st
import google.generativeai as genai
import whisper
import torch
import tempfile
import os
import base64
import sounddevice as sd
import numpy as np
import wave
from gtts import gTTS  

# Configure Gemini API
genai.configure(api_key="google_api_key") 


# Load Whisper model
whisper_model = whisper.load_model("base")

# Streamlit UI
st.title("🎙️ Interactive Voice Chatbot (Whisper + Gemini Pro)")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "model", "content": "Hello! How can I help you today?"}
    ]

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    st.write("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    st.write("Recording complete!")

    # Save as WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        with wave.open(temp_audio, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return temp_audio.name

# Function to get response from Gemini while maintaining chat history
def get_gemini_response():
    model = genai.GenerativeModel("gemini-pro")
    
    # Format conversation history correctly (Gemini expects "user" & "model" roles)
    conversation = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            conversation.append({"role": "user", "parts": [msg["content"]]})
        else:
            conversation.append({"role": "model", "parts": [msg["content"]]})

    # Get response from Gemini
    response = model.generate_content(conversation)
    return response.text  # Extract text response

# Display chat history (including human and LLM responses)
st.subheader("🗨️ Chat History")
for msg in st.session_state.messages:
    role = "🤖" if msg["role"] == "model" else "🗣️"
    st.write(f"{role} **{msg['content']}**")

# Button to start recording
if st.button("🎤 Record & Send Voice"):
    # Record audio and process it
    audio_path = record_audio()
    
    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_path)
    user_text = result["text"]

    # Add user message (transcription) to chat history
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Get AI response
    bot_text = get_gemini_response()
    st.session_state.messages.append({"role": "model", "content": bot_text})

    # Display the transcription and response
    st.subheader("🗨️ Updated Chat History")
    for msg in st.session_state.messages:
        role = "🤖" if msg["role"] == "model" else "🗣️"
        st.write(f"{role} **{msg['content']}**")

    # Convert bot response to speech using gTTS
    tts = gTTS(text=bot_text, lang="en")
    audio_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_output.name)

    # Auto-play response (removes need for clicking play)
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mpeg;base64,{base64.b64encode(open(audio_output.name, "rb").read()).decode()}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Button to Reset Chat
if st.button("🔄 Reset Chat"):
    st.session_state.messages = [{"role": "model", "content": "Hello! How can I help you today?"}]
    st.rerun()


# run command - streamlit run problem_1.py
# I have used chatgpt for the generation of the code, Prompts have been refined multiple times such as changing the llm model used, the tts model etc. The streamlit ui was created so that it can be used to interact rather than a fastAPI end point as there was no Frontend available. This could be made better with more resources and time but as it an evaluation I did not want to take a lot of time.






# THE BELOW PROVIDED IS THE SOLUTION USING FASTAPI, I HAVE COMMENTED IT AS I COULD NOT TEST IT 


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import google.generativeai as genai
# import whisper
# import tempfile
# import os
# import base64
# import sounddevice as sd
# import numpy as np
# import wave
# from gtts import gTTS  # Free Online TTS
# from fastapi.responses import HTMLResponse

# # Configure Gemini API
# genai.configure(api_key="google_api_key")

# # Load Whisper model
# whisper_model = whisper.load_model("base")

# # FastAPI instance
# app = FastAPI()

# # Data model for audio response request
# class AudioRequest(BaseModel):
#     duration: int = 5  # default duration for recording in seconds

# # Function to record audio
# def record_audio(duration=5, sample_rate=16000):
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
#     sd.wait()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#         with wave.open(temp_audio, 'wb') as wf:
#             wf.setnchannels(1)
#             wf.setsampwidth(2)
#             wf.setframerate(sample_rate)
#             wf.writeframes(audio_data.tobytes())
#         return temp_audio.name

# # Function to get response from Gemini while maintaining chat history
# def get_gemini_response(messages):
#     model = genai.GenerativeModel("gemini-pro")
#     conversation = [{"role": "user", "parts": [msg["content"]]} if msg["role"] == "user" else {"role": "model", "parts": [msg["content"]]} for msg in messages]
#     response = model.generate_content(conversation)
#     return response.text

# # Endpoint to start and stop recording
# @app.post("/record_audio")
# async def record_audio_endpoint(request: AudioRequest):
#     audio_path = record_audio(duration=request.duration)
    
#     # Transcribe with Whisper
#     result = whisper_model.transcribe(audio_path)
#     user_text = result["text"]
    
#     # Get AI response
#     bot_text = get_gemini_response([{"role": "user", "content": user_text}])
    
#     # Convert bot response to speech using gTTS
#     tts = gTTS(text=bot_text, lang="en")
#     audio_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(audio_output.name)

#     # Return audio response as base64
#     with open(audio_output.name, "rb") as audio_file:
#         audio_base64 = base64.b64encode(audio_file.read()).decode()

#     return {
#         "transcription": user_text,
#         "bot_response": bot_text,
#         "audio_base64": audio_base64
#     }

# # Endpoint to reset chat (no state in this case for simplicity)
# @app.post("/reset_chat")
# async def reset_chat():
#     return {"message": "Chat has been reset."}

# # Main page (serving HTML)
# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     html_content = """
#     <html>
#         <head>
#             <title>Voice Chatbot (Whisper + Gemini)</title>
#         </head>
#         <body>
#             <h2>Welcome to the Voice Chatbot!</h2>
#             <p>Use the /record_audio endpoint to start the conversation. Send an audio request.</p>
#             <p>Use /reset_chat to reset the chat.</p>
#         </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)


# run command - uvicorn problem_1:app --reload



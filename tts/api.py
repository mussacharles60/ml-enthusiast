from flask import Flask, request, jsonify
from transformers import VitsModel, AutoModelForTextToWaveform, AutoTokenizer
from pydub import AudioSegment
from io import BytesIO
import base64
import torch
import scipy
import numpy as np
import noisereduce as nr

app = Flask(__name__)

# Load the Hugging Face TTS model and processor (once at startup)
model_en = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer_en = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

model_sw = VitsModel.from_pretrained("facebook/mms-tts-swh")
tokenizer_sw = AutoTokenizer.from_pretrained("facebook/mms-tts-swh")

# model_sw = AutoModelForTextToWaveform.from_pretrained("khof312/mms-tts-swh-female-2")
# tokenizer_sw = AutoTokenizer.from_pretrained("khof312/mms-tts-swh-female-2")

# Function to reduce noise from audio
def reduce_noise(audio_data, sample_rate):
    # Apply noise reduction using the noisereduce library
    reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    return reduced_audio

# Helper function to generate speech from text using Hugging Face model
def text_to_speech(text, lang):
    
    output = None

    # Generate speech audio from text
    if lang == "en":
      inputs = tokenizer_en(text, return_tensors="pt")
      with torch.no_grad():
        output = model_en(**inputs).waveform
    elif lang == "sw":
      inputs = tokenizer_sw(text, return_tensors="pt")
      with torch.no_grad():
        output = model_sw(**inputs).waveform

    if output == None:
      return None
    
    # Convert PyTorch tensor to NumPy array
    # audio_array = output.squeeze().cpu().numpy()
    audio_array = output.detach().cpu().numpy().squeeze()
    audio_array = (audio_array * 32767).astype(np.int16)  # scale to int16
    audio_array = audio_array.astype("int16")  # Convert to int16 for audio

    # Reduce noise from the recorded audio
    audio_array = reduce_noise(audio_array, sample_rate=16000)

    # # scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=output.float().numpy())
    scipy.io.wavfile.write("output.wav", rate=16000, data=audio_array)

    # Convert to numpy array for manipulation in pydub
    # audio_array = speech.detach().numpy().astype("int16")  # Convert to int16 for audio
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=16000,  # Assuming the sample rate is 16kHz
        sample_width=2,  # 16-bit audio = 2 bytes
        channels=1  # Mono audio
    )

    return audio_segment

# API route to accept text input and return Base64-encoded audio
@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json  # Expecting JSON payload with 'text' field
    text = data.get("text", "")
    lang = data.get("lang", "")

    if not text or not lang:
        return jsonify({
          "err": {
            "code": 402,
            "msg": "A valid text and language are required"
          }
        }), 402

    # Step 1: Generate speech from text
    audio = text_to_speech(text, lang)

    if audio == None:
      return jsonify({
          "err": {
            "code": 500,
            "msg": "Internal server error. Unable to generate audio"
          }
        }), 500

    # Step 2: Apply pitch modification (increase pitch by 20%)
    new_sample_rate = int(audio.frame_rate * 1.0)
    pitched_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})

    # Set the frame rate back to the original value for playback
    pitched_audio = pitched_audio.set_frame_rate(audio.frame_rate)

    # Save the pitch-shifted audio
    pitched_audio.export("output_pitch_shifted_pydub.wav", format="wav")

    # Step 3: Convert the audio to bytes
    buffer = BytesIO()
    pitched_audio.export(buffer, format="wav")
    audio_bytes = buffer.getvalue()

    # Step 4: Encode the audio as base64
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_data_uri = f"data:audio/wav;base64,{audio_base64}"

    # Step 5: Return the Base64-encoded audio as a JSON response
    return jsonify({"audio": audio_data_uri}), 200

if __name__ == '__main__':
    app.run(debug=True, port=3001)

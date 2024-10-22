import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from flask import Flask, request, jsonify
from transformers import VitsModel, AutoTokenizer
from pydub import AudioSegment
from io import BytesIO
import base64
import torch
import scipy
import numpy as np
import noisereduce as nr
import re

app = Flask(__name__)

# Load the Hugging Face TTS model and processor (once at startup)

# model_en = VitsModel.from_pretrained("BHOSAI/SARA_TTS")
# tokenizer_en = AutoTokenizer.from_pretrained("BHOSAI/SARA_TTS")

# model_en = VitsModel.from_pretrained("facebook/mms-tts-eng")
# tokenizer_en = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

model_en = VitsModel.from_pretrained("ylacombe/vits_ljs_welsh_female_monospeaker_2")
tokenizer_en = AutoTokenizer.from_pretrained('ylacombe/vits_ljs_welsh_female_monospeaker_2')

# model_sw = VitsModel.from_pretrained("facebook/mms-tts-swh")
# tokenizer_sw = AutoTokenizer.from_pretrained("facebook/mms-tts-swh")

# model_sw = AutoModelForTextToWaveform.from_pretrained("khof312/mms-tts-swh-female-2")
# tokenizer_sw = AutoTokenizer.from_pretrained("khof312/mms-tts-swh-female-2")

model_sw = VitsModel.from_pretrained("mussacharles60/swahili-tts-female-voice")
tokenizer_sw = AutoTokenizer.from_pretrained("mussacharles60/swahili-tts-female-voice")


def clean_text(input_text):
    """
    Remove special characters from the input text, keeping only commas and periods.

    :param input_text: The original input text.
    :return: Cleaned text with only allowed characters.
    """
    # Remove special characters except for comma and period
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,\"\'><+=-_?@#$%!*& ]+', '', input_text)
    return cleaned_text

# Function to reduce noise from audio
def reduce_noise(audio_data, sample_rate):
    # Apply noise reduction using the noisereduce library
    reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    return reduced_audio

def generate_audio_segment(text: str, lang: str):
    print(f"[INFO] ---- processing input_text: ", text)
    output = None
    sample_rate = 16000
    audio_array = None

    # Generate speech audio from text
    if lang == "en":
        inputs = tokenizer_en(text, return_tensors="pt")
        with torch.no_grad():
            output = model_en(**inputs).waveform

        sample_rate = model_en.config.sampling_rate

        # Convert PyTorch tensor to NumPy array
        audio_array = output.detach().cpu().numpy().squeeze()

        # TODO: Trim the first 1 second of the generated audio, only for this model = "ylacombe/vits_ljs_welsh_female_monospeaker_2"
        audio_array = audio_array[int(sample_rate * 1):]  # Remove the first 1 second

    elif lang == "sw":
        inputs = tokenizer_sw(text, return_tensors="pt")
        with torch.no_grad():
            output = model_sw(**inputs).waveform
        
        sample_rate = model_sw.config.sampling_rate

        # Convert PyTorch tensor to NumPy array
        audio_array = output.detach().cpu().numpy().squeeze()

    else:
        return None, 0
   
    # Convert PyTorch tensor to NumPy array
    # audio_array = output.squeeze().cpu().numpy()
    # audio_array = output.detach().cpu().numpy().squeeze()
    audio_array = (audio_array * 32767).astype(np.int16)  # scale to int16
    audio_array = audio_array.astype("int16")  # Convert to int16 for audio

    return audio_array, sample_rate


# def generate_audio_from_text(input_text: str, lang: str):
#     # Step 1: Split the input text by full stop '.'
#     text_segments = [segment.strip() for segment in input_text.split('.') if segment.strip()]

#     # Step 2: Generate a short blank audio (silence) for pauses
#     pause_duration=0.5
#     sampling_rate=22050
#     blank_audio = np.zeros(int(sampling_rate * pause_duration), dtype=np.float32)

#     # Step 3: Generate audio for each text segment and concatenate them with pauses
#     audio_segments = []
#     sample_rate = 16000
#     i = 0
#     for segment in text_segments:
#         i += 1
#         if segment:  # Skip empty segments
#             # Step 4: Generate audio for the current text segment
#             audio_array, sample_rate = generate_audio_segment(segment, lang)  # Replace with your TTS audio generation function
#             scipy.io.wavfile.write(f"segments/output-segment-{lang}-{i}.wav", rate=sample_rate, data=audio_array)
#             print(f"segment {i}: sample rate: {sample_rate}, audio: ", audio_array)
#             audio_segments.append(audio_array)

#             # Get the dtype of the generated audio
#             audio_dtype = audio_array.dtype

#             # Append the silence after the segment
#             blank_audio = np.zeros(int(sample_rate * pause_duration), dtype=audio_dtype)
            
#             audio_segments.append(blank_audio)

#     # Step 5: Concatenate all audio segments
#     final_audio = np.concatenate(audio_segments[:-1])  # Exclude the last silence

#     return final_audio, sample_rate


def generate_audio_from_text(input_text: str, lang: str):
    # Step 1: Split the input text by full stop (.) and handle commas (,) separately
    text_segments = []
    for segment in input_text.split('. '):
        # Further split by commas
        comma_segments = [sub_segment.strip() for sub_segment in segment.split(', ') if sub_segment.strip()]
        text_segments.append(comma_segments)

    # Step 2: Generate short blank audio (silence) for pauses
    full_stop_blank_audio = None
    comma_blank_audio = None

    full_stop_pause_duration = 0.1
    comma_pause_duration = 0.01

    # Step 3: Generate audio for each text segment and concatenate them with pauses
    sample_rate = 16000

    audio_segments = []
    i = 0
    for segment_list in text_segments:
        for idx, segment in enumerate(segment_list):
            i += 1
            if segment:  # Skip empty segments
                # Step 4: Generate audio for the current text segment
                audio_array, sample_rate = generate_audio_segment(segment, lang)

                if sample_rate != 0:
                    # scipy.io.wavfile.write(f"segments/output-segment-{lang}-{i}.wav", rate=sample_rate, data=audio_array)
                    # print(f"segment {i}: sample rate: {sample_rate}, audio: ", audio_array)

                    # Get the dtype of the generated audio
                    audio_dtype = audio_array.dtype

                    # Create blank audios if not already created
                    if full_stop_blank_audio is None:
                        full_stop_blank_audio = np.zeros(int(sample_rate * full_stop_pause_duration), dtype=audio_dtype)
                    if comma_blank_audio is None:
                        comma_blank_audio = np.zeros(int(sample_rate * comma_pause_duration), dtype=audio_dtype)

                    audio_segments.append(audio_array)

                    # Add comma pause unless it's the last segment
                    if idx < len(segment_list) - 1:
                        audio_segments.append(comma_blank_audio)

        # Append the silence after the entire segment (full stop pause)
        audio_segments.append(full_stop_blank_audio)
    
    # Step 5: Concatenate all audio segments
    final_audio = np.concatenate(audio_segments[:-1])  # Exclude the last silence

    return final_audio, sample_rate


# Helper function to generate speech from text using Hugging Face model
def text_to_speech(input_text: str, lang: str):
    input_text = clean_text(input_text)

    final_audio, sample_rate = generate_audio_from_text(input_text, lang)
    audio_array = final_audio

    # Reduce noise from the recorded audio
    # audio_array = reduce_noise(audio_array, sample_rate)

    # # scipy.io.wavfile.write("output.wav", rate=model.config.sampling_rate, data=output.float().numpy())
    # scipy.io.wavfile.write("output.wav", rate=suno_model.generation_config.sample_rate, data=audio_array)
    scipy.io.wavfile.write(f"output-{lang}.wav", rate=sample_rate, data=audio_array)

    # Convert to numpy array for manipulation in pydub
    # audio_array = speech.detach().numpy().astype("int16")  # Convert to int16 for audio
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,  # Assuming the sample rate is 16kHz
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
    pitched_audio = audio._spawn(audio.raw_data, overrides={
                                 'frame_rate': new_sample_rate})

    # Set the frame rate back to the original value for playback
    pitched_audio = pitched_audio.set_frame_rate(audio.frame_rate)

    # Save the pitch-shifted audio
    pitched_audio.export(f"output_pitch_shifted_pydub-{lang}.wav", format="wav")

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

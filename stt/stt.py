from speechbrain.inference.ASR import EncoderASR
import pyaudio
import numpy as np
import wave
import os
import noisereduce as nr
# import matplotlib.pyplot as plt
import keyboard
# from IPython.display import Audio, display
# import time

# define automatic speech recognition model
asr_model = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-dvoice-swahili",
    savedir="pretrained_models/asr-wav2vec2-dvoice-swahili"
)

# define temp audio file path
output_dir = 'input_audios'
os.makedirs(output_dir, exist_ok=True)

# Get current time in milliseconds (since epoch)
# current_time_ms = str(int(time.time() * 1000))
current_time_ms = '0'

# Define the output file path
output_file = os.path.join(output_dir, current_time_ms + '-og.wav')


def select_audio_device():
    # List available input audio devices and their names
    global p
    p = pyaudio.PyAudio()
    print('---------------------------------------------------')
    print('Available input devices:')
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f'{i}: {device_info["name"]}')

    # Select the desired input device by its ID (replace with the appropriate ID)
    global selected_device_id
    selected_device_id = 8  # Change this to the ID of your desired input device

    print(f'Select available device by number: 0 - {p.get_device_count() - 1} or ESC to exit')
    print('---------------------------------------------------')
    while True:
      if keyboard.read_key() == '0':
          selected_device_id = 0
          break
      elif keyboard.read_key() == '1':
          selected_device_id = 1
          break
      elif keyboard.read_key() == '2':
          selected_device_id = 2
          break
      elif keyboard.read_key() == '3':
          selected_device_id = 3
          break
      elif keyboard.read_key() == '4':
          selected_device_id = 4
          break
      elif keyboard.read_key() == '5':
          selected_device_id = 5
          break
      elif keyboard.read_key() == '6':
          selected_device_id = 6
          break
      elif keyboard.read_key() == '7':
          selected_device_id = 7
          break
      elif keyboard.read_key() == '8':
          selected_device_id = 8
          break
      elif keyboard.read_key() == '9':
          selected_device_id = 9
          break
      elif keyboard.read_key() == 'Esc':
          exit(0)
          break
    # Record audio from the selected input device
    print(
        f'Recording audio will use device: "{p.get_device_info_by_index(selected_device_id)["name"]}"')
    print('---------------------------------------------------')


# Function to start recording
def start_recording():
    global recording
    recording = True
    print('Recording audio. Press the space key to stop...')


# Function to stop recording
def stop_recording():
    global recording
    recording = False
    print('Recording stopped.')


# Function to reduce noise from audio
def reduce_noise(audio_data, sample_rate):
    # Apply noise reduction using the noisereduce library
    reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    return reduced_audio


# Callback function for audio recording
def audio_callback(in_data, frame_count, time_info, status):
    global recording
    global audio
    if recording:
        audio.append(in_data)
    return (in_data, pyaudio.paContinue)


def recording_loop():
    select_audio_device()
    global p
    global selected_device_id
    global audio
    # Set the parameters for audio recording
    global sample_rate
    sample_rate = 16000  # You can adjust this based on your needs
    duration = 10        # Maximum recording duration in seconds

    # Initialize recording status
    global recording
    recording = False
    audio = []

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open an audio stream for recording
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    input_device_index=selected_device_id,
                    frames_per_buffer=1024,
                    stream_callback=audio_callback)

    # Wait for the space key to start recording
    print('Press the space key to start recording...')
    keyboard.wait('space')
    start_recording()

    # Wait for the space key to stop recording
    # print('Recording audio. Press the space key to stop...')
    keyboard.wait('space')
    stop_recording()
    print('---------------------------------------------------')

    # Close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    process_audio(audio)


def process_audio(audio):
    global sample_rate
    # print(f'process_audio: {audio}')
    # Save the recorded audio as a WAV file
    if audio:
        audio_data = b''.join(audio)
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        print(f'Audio saved as {output_file}')

        # Reduce noise from the recorded audio
        if os.path.exists(output_file):
            with wave.open(output_file, 'rb') as wf:
                audio_data = wf.readframes(-1)
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                sample_rate = wf.getframerate()

            # Reduce noise from the recorded audio
            reduced_audio = reduce_noise(audio_data, sample_rate)

            # Save the reduced audio
            reduced_output_file = os.path.join(
                output_dir, current_time_ms + '-clean.wav')
            with wave.open(reduced_output_file, 'wb') as wf:
                wf.setnchannels(1)  # Mono audio
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(reduced_audio.tobytes())

            print(f'Noise reduced audio saved as {reduced_output_file}')

        # Load the recorded audio file
        audio_file = output_dir + '/' + current_time_ms + '-og.wav'

        # Load the reduced audio file
        reduced_audio_file = output_dir + '/' + current_time_ms + '-clean.wav'
        voice_to_text(reduced_audio_file)

    else:
        print('No audio recorded.')
        end()


def voice_to_text(audio_path=''):
    print('---------------------------------------------------')
    # Try to recognize the word by using asr_model
    print('decoding text...')
    decoded_text = asr_model.transcribe_file(
        audio_path
    )
    print(f'decoded text: {decoded_text}')
    print('---------------------------------------------------')
    end()


def end():
    # Wait for the space key to start recording
    print('Press the space key to start again or ESC to exit')
    while True:
      if keyboard.read_key() == 'space':
          recording_loop()
          break
      elif keyboard.read_key() == 'Esc':
          exit(0)
          break
      # else:
      #     end()
      #     break


if __name__ == '__main__':
    recording_loop()

import serial
import numpy as np
import wave
import pyaudio

CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

# Open serial port connection to Arduino
ser = serial.Serial('COM3', 115200)

# Read audio data from Arduino
audio_data = []
for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
    raw_data = ser.read(2)  # Read 2 bytes (16 bits) per sample
    sample = int.from_bytes(raw_data, byteorder='little', signed=True)
    audio_data.append(sample)

# Close serial port connection
ser.close()

# Save audio data as a .wav file
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(RATE)
    wf.writeframes(np.array(audio_data))

print("Recording saved as", WAVE_OUTPUT_FILENAME)


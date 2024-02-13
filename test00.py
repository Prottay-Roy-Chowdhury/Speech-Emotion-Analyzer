# import numpy as np
# import scipy.io.wavfile as wavfile
# import matplotlib.pyplot as plt

# # Read the audio file
# sampling_rate, data = wavfile.read('output10.wav')

# # Extract one channel if it's stereo
# if len(data.shape) > 1:
#     data = data[:, 0]

# # Compute the one-dimensional discrete Fourier Transform
# fft_result = np.fft.fft(data)

# # Calculate the frequencies corresponding to the FFT result
# frequencies = np.fft.fftfreq(len(fft_result), 1 / sampling_rate)

# # Plot the frequency spectrum
# plt.figure(figsize=(10, 5))
# plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(fft_result)//2])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title('Frequency Spectrum')
# plt.grid(True)
# plt.show()

# ##############################################

# import scipy.io.wavfile as wavfile
# import numpy as np

# # Read the audio file
# sampling_rate, data = wavfile.read('output10.wav')

# # Extract one channel if it's stereo
# if len(data.shape) > 1:
#     data = data[:, 0]

# # Normalize the amplitude values to the range [-1, 1]
# normalized_data = data / np.max(np.abs(data))

# # Plot the amplitude over time
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(len(normalized_data)) / sampling_rate, normalized_data)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Amplitude vs Time')
# plt.grid(True)
# plt.show()

#############################################

# import scipy.io.wavfile as wavfile
# import numpy as np

# # Read the audio file
# sampling_rate, data = wavfile.read('output15.wav')

# # Extract one channel if it's stereo
# if len(data.shape) > 1:
#     data = data[:, 0]

# # Calculate RMS amplitude
# rms_amplitude = np.sqrt(np.mean(data ** 2))

# # Calculate peak amplitude
# peak_amplitude = np.max(np.abs(data))

# if rms_amplitude >= 50:
#     print("turn left")
# else:
#     print("turn right")


# print("RMS Amplitude:", rms_amplitude)
# print("Peak Amplitude:", peak_amplitude)


# # Calculate maximum and minimum amplitude
# max_amplitude = np.max(data)
# min_amplitude = np.min(data)

# # Calculate average of maximum and minimum amplitude
# average_amplitude = (max_amplitude + min_amplitude) / 2

# print("Maximum Amplitude:", max_amplitude)
# print("Minimum Amplitude:", min_amplitude)
# print("Average of Maximum and Minimum Amplitude:", average_amplitude)


#############################################


import numpy as np
import pyaudio

# Parameters for audio stream
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 44100  # Sample rate in Hz
CHUNK = 1024  # Number of frames per buffer

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* Recording live audio...")

try:
    while True:
        # Read audio data from stream
        data = stream.read(CHUNK)
        # Convert binary data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Process audio data here
        # For example, calculate RMS amplitude
        rms_amplitude = np.sqrt(np.mean(audio_data ** 2))
        
        print("RMS Amplitude:", rms_amplitude)
        
        if rms_amplitude >= 50:
            print("turn left")
        else:
            print("turn right")

except KeyboardInterrupt:
    print("* Stopped recording")

# Close stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()

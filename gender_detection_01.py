import librosa
import librosa.display
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
from matplotlib.pyplot import specgram


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
# from keras.optimizers.legacy import Adam
# #Define the learning rate
# learning_rate = 0.001  # Example learning rate

# # Adjust other hyperparameters as needed
# opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# # score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
# # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



# #### The file 'output10.wav' in the next cell is the file that was recorded live using the code in AudioRecoreder notebook found in the repository

data, sampling_rate = librosa.load('output17.wav')

import pandas as pd

# plt.figure(figsize=(15, 5))
# librosa.display.waveshow(data, sr=sampling_rate)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Waveform')
# plt.show()

#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('output17.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T

livedf2

twodim= np.expand_dims(livedf2, axis=2)

livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)

livepreds

livepreds1=livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()

#livepredictions = (lb.inverse_transform((liveabc)))
#livepredictions
from sklearn.preprocessing import LabelBinarizer

# Assuming liveabc contains encoded predictions
# Instantiate LabelBinarizer
lb = LabelBinarizer()

# Fit LabelBinarizer to your data (assuming you have labels in liveabc)
lb.fit(liveabc)

# Use LabelBinarizer to inverse-transform labels
#livepredictions = lb.inverse_transform(liveabc)

livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions)

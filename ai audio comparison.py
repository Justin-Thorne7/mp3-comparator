# Import the necessary libraries
import librosa
from tkinter import filedialog
from tkinter import Tk
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib.pyplot as plt 

# Define the compare_audio_clips function
def compare_audio_clips(clip1, clip2):
  # Compare the two audio clips and return their similarity as a percentage

  #extract the audio features from the two clips

  def extract_features(audio):
    # Use Librosa to extract the audio features
    features = librosa.feature.mfcc(audio)

    # Flatten the features into a single vector
    features = features.flatten()

    #features = features[int(0.3*len(features)):]

    return features

  features1 = extract_features(clip1)
  features2 = extract_features(clip2)


  # Normalize the length of the vectors by padding the shorter vector with zeros
  if features1.shape[0] > features2.shape[0]:
    features2 = np.pad(features2, (0, features1.shape[0] - features2.shape[0]), 'constant')
  elif features2.shape[0] > features1.shape[0]:
    features1 = np.pad(features1, (0, features2.shape[0] - features1.shape[0]), 'constant')

  # Next, we need to calculate the similarity between the two sets of features
  # using the Euclidean distance between the two sets of features as a measure of similarity
  distance = euclidean(features1, features2)

  # We can then convert the distance into a similarity percentage by taking the inverse of the distance and normalizing it between 0 and 100
  similarity = 100 / (1 + distance)


  return similarity

# Open a file explorer to allow the user to select the first audio file
root = Tk()
root.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("MP3 files","*.mp3"),("all files","*.*")))

try:
  # Load the selected audio file using Librosa
  clip1, _ = librosa.load(root.filename)
except Exception as e:
  print('Error: Failed to load the first audio file')
  print(e)

# Open a file explorer to allow the user to select the second audio file
root = Tk()
root.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("MP3 files","*.mp3"),("all files","*.*")))

try:
  # Load the selected audio file using Librosa
  clip2, _ = librosa.load(root.filename)
except Exception as e:
  print('Error: Failed to load the second audio file')
  print(e)

similarity = compare_audio_clips(clip1, clip2)
print('The similarity between the two audio clips is: {}%'.format(similarity*10))

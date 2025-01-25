import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import whisper
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import joblib

class AudioFeatureExtractor:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.transcription_model = whisper.load_model("base")
        self.acoustic_features = None
        self.linguistic_features = None
        self.combined_features = None
        # Load the SVM model
        self.model = joblib.load("svm_model.pkl")
    
    def extract_acoustic_features(self):
        y, sr = librosa.load(self.audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
        jitter = np.mean(librosa.feature.rms(y=y))
        shimmer = np.mean(librosa.effects.harmonic(y))

        self.acoustic_features = {
            "MFCC_1": mfccs[0], "MFCC_2": mfccs[1], "MFCC_3": mfccs[2],
            "Spectral_Centroid": spectral_centroid,
            "Spectral_Rolloff": spectral_rolloff,
            "Zero_Crossing_Rate": zero_crossing_rate,
            "Chroma_STFT": chroma_stft,
            "Jitter": jitter,
            "Shimmer": shimmer
        }
        return self.acoustic_features

    def transcribe_audio(self):
        y, sr = librosa.load(self.audio_file, sr=None)
        result = self.transcription_model.transcribe(y)
        return result["text"]

    def extract_linguistic_features(self, text):
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        token_count = len(words)
        fillers = ['um', 'uh', 'er', 'hmm', 'like', 'you know', 'actually', 'so']
        filler_count = sum(1 for word in words if word.lower() in fillers)
        type_count = len(set(words))
        type_token_ratio = type_count / token_count if token_count > 0 else 0
        window_size = 50
        ma_ttr = np.mean([len(set(words[i:i + window_size])) / len(words[i:i + window_size]) 
                          for i in range(0, len(words), window_size)]) if len(words) > window_size else type_token_ratio
        N = len(words)
        V = len(set(words))
        brunets_index = (V - 1) / (N - V) if N > 1 and V > 1 else 0
        stop_words = set(stopwords.words('english'))
        content_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        content_density = len(content_words) / token_count if token_count > 0 else 0
        sentence_count = len(sentences)
        avg_words_per_sentence = token_count / sentence_count if sentence_count > 0 else 0

        self.linguistic_features = {
            "filler_count": filler_count,
            "token_count": token_count,
            "type_token_ratio": type_token_ratio,
            "ma_ttr": ma_ttr,
            "brunets_index": brunets_index,
            "content_density": content_density,
            "average_words_per_sentence": avg_words_per_sentence,
            "sentence_count": sentence_count
        }
        return self.linguistic_features

    def process_audio_file(self):
        acoustic_features = self.extract_acoustic_features()
        transcription = self.transcribe_audio()
        linguistic_features = self.extract_linguistic_features(transcription)
        self.combined_features = {**acoustic_features, **linguistic_features}
        return self.combined_features

    def predict(self):
        if self.combined_features is None:
            self.process_audio_file()
        feature_vector = np.array(list(self.combined_features.values())).reshape(1, -1)
        prediction = self.model.predict(feature_vector)
        if prediction[0] == 1:
            return "Yes"
        else:
            return "No"


# Example Usage
if __name__ == "__main__":
    audio_file = r'F:\Chatbox_dementia\uploads\Process-test-009__SFT.wav'
    feature_extractor = AudioFeatureExtractor(audio_file)
    features = feature_extractor.process_audio_file()
    print("Extracted Features:", features)
    prediction = feature_extractor.predict()
    print("Prediction:", prediction)

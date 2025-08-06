# gemma_loader.py
import os
import torch
import joblib
import tensorflow as tf
from unsloth import FastModel
from transformers import TextStreamer
import google.generativeai as genai
import numpy as np
import librosa

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],), initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, axis=-1)
        return tf.reduce_sum(inputs * a, axis=1)

class MultimodalGemmaLoader:
    def __init__(self, model_name="unsloth/gemma-3n-E4B-it"):
        self.model_name = model_name
        self.model, self.tokenizer = None, None
        self.emotion_model, self.label_encoder = None, None
        self.tts_model = None

    def load_gemma_model(self):
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name, dtype=None, max_seq_length=1024,
            load_in_4bit=True, full_finetuning=False,
        )

    def load_emotion_model(self, model_path='emotion_model.h5', encoder_path='label_encoder.joblib'):
        try:
            with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
                self.emotion_model = tf.keras.models.load_model(model_path)
            self.label_encoder = joblib.load(encoder_path)
        except Exception as e:
            print(f"Warning: Could not load emotion model: {e}")

    def setup_tts(self, api_key):
        genai.configure(api_key=api_key)
        self.tts_model = genai.GenerativeModel('gemini-2.5-flash-preview-tts')

    def transcribe_audio(self, audio_data):
        messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_data}, {"type": "text", "text": "Transcribe this audio accurately."}]}]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return response.split("assistant")[-1].strip()

    def analyze_visual_emotion(self, image_data):
        messages = [{"role": "user", "content": [{"type": "image", "image": image_data}, {"type": "text", "text": "Analyze the emotion shown in this person's face. Return only one word: happy, sad, angry, fearful, surprised, disgusted, neutral, or calm."}]}]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=64)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        torch.cuda.empty_cache()
        emotion = response.split("assistant")[-1].strip().lower()
        valid_emotions = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted', 'neutral', 'calm']
        for e_word in valid_emotions:
            if e_word in emotion: return e_word
        return 'neutral'

    def extract_mfcc_single(self, audio_path):
        a, sr = librosa.load(audio_path, sr=16000)
        m = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=40)
        p = 216 - m.shape[1]
        if p > 0: m = np.pad(m, ((0,0),(0,p)), mode='constant')
        else: m = m[:,:216]
        return m.T

    def predict_vocal_emotion(self, audio_path):
        if self.emotion_model is None or self.label_encoder is None: return "neutral"
        mfcc = self.extract_mfcc_single(audio_path)
        mfcc = np.expand_dims(np.expand_dims(mfcc, -1), 0)
        pred = self.emotion_model.predict(mfcc)
        return self.label_encoder.inverse_transform([np.argmax(pred)])[0]

    def generate_tts(self, text, voice_config=None):
        if not self.tts_model: return None
        try:
            response = self.tts_model.generate_content(text, generation_config={"response_modalities": ["audio"], "speech_config": voice_config or {"voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}}})
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

def load_multimodal_system():
    loader = MultimodalGemmaLoader()
    loader.load_gemma_model()
    loader.load_emotion_model()
    return loader
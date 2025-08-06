# train_and_save.py
import os, numpy as np, librosa, joblib, logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- GPU CONFIGURATION ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.info(f"Using GPU: {gpus[0]}")
    else:
        logging.warning("No GPU devices found. TensorFlow will run on CPU.")
except Exception as e:
    logging.error(f"Error configuring GPU: {e}. Falling back to CPU.")

# --- CONFIGURATION ---
DATA_DIR = 'data'
RAVDESS_PATH = DATA_DIR
MAX_PAD_LEN, N_MFCC = 216, 40
SAMPLE_RATE = 16000

# --- DATA AUGMENTATION FUNCTIONS ---
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def apply_pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def stretch(data, rate=1.1):
    return librosa.effects.time_stretch(y=data, rate=rate)

def time_shift(data, shift_max=0.2):
    shift = np.random.randint(int(len(data) * shift_max))
    return np.roll(data, shift)

# --- FEATURE EXTRACTION ---
def extract_mfcc(file_path, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC, sr=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        aug_data = [audio, add_noise(audio), apply_pitch_shift(audio, sr), stretch(audio), time_shift(audio)]
        mfccs = []
        for signal in aug_data:
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
            p = max_pad_len - mfcc.shape[1]
            if p > 0: mfcc = np.pad(mfcc, ((0,0),(0,p)), mode='constant')
            else: mfcc = mfcc[:,:max_pad_len]
            mfccs.append(mfcc.T)
        return mfccs
    except Exception as e:
        logging.warning(f"Skipping {file_path} due to error: {e}")
        return []

logging.info("Loading data with augmentation...")
X, y = [], []
emotion_map = {
    'a': 'angry', 'd': 'disgust', 'f': 'fearful', 'h': 'happy',
    'n': 'neutral', 'sa': 'sad', 'su': 'surprised', 'c': 'calm'
}

if not os.path.isdir(RAVDESS_PATH):
    logging.error(f"Dataset path does not exist: {RAVDESS_PATH}. Please ensure your 'data' folder contains the .wav files directly.")
    raise FileNotFoundError(f"Dataset path does not exist: {RAVDESS_PATH}")

for f_name in os.listdir(RAVDESS_PATH):
    if f_name.lower().endswith('.wav'):
        file_path = os.path.join(RAVDESS_PATH, f_name)
        mfcc_list = extract_mfcc(file_path)
        if mfcc_list:
            parts = f_name.split('_')
            if len(parts) > 1:
                base_name_without_ext = parts[1].split('.')[0]
                emotion_code = ''.join(c for c in base_name_without_ext if c.isalpha())
                if emotion_code in emotion_map:
                    for mfcc in mfcc_list:
                        X.append(mfcc)
                        y.append(emotion_map[emotion_code])
                else:
                    logging.warning(f"Unknown emotion code '{emotion_code}' in file '{f_name}'. Skipping.")
            else:
                logging.warning(f"Could not parse emotion from filename '{f_name}'. Skipping.")

if not X:
    raise ValueError("No features extracted. Check dataset path and file naming convention.")

X, y = np.array(X), np.array(y)
X = np.expand_dims(X, -1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
num_classes = y_cat.shape[1]
X_train,X_test,y_train,y_test=train_test_split(X,y_cat,test_size=0.2,random_state=42,stratify=y_cat)

logging.info(f"Loaded {len(X)} samples (including augmented). Training data shape: {X_train.shape}")


# --- Mixup Augmentation for Training ---
def mixup(batch_x, batch_y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = tf.shape(batch_x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_x = lam * batch_x + (1 - lam) * tf.gather(batch_x, index)
    mixed_y = lam * batch_y + (1 - lam) * tf.gather(batch_y, index)
    return mixed_x, mixed_y

batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1024).batch(batch_size)
train_ds = train_ds.map(lambda x, y: mixup(x, y, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# --- ATTENTION LAYER ---
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, axis=-1)
        return tf.reduce_sum(inputs * a, axis=1)

# --- BUILD MODEL ---
def build_model(shape, classes):
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(i)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)
    x = Reshape((shape[0], 5 * 128))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.4)(x)
    x = AttentionLayer()(x)
    m = Model(inputs=i, outputs=Dense(classes, activation='softmax')(x))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m

model = build_model((MAX_PAD_LEN, N_MFCC, 1), num_classes)
cbs=[EarlyStopping(patience=10,restore_best_weights=True),ReduceLROnPlateau(patience=5)]
logging.info("Training model...")
model.fit(train_ds, validation_data=test_ds, epochs=100, callbacks=cbs)
model.save('emotion_model.h5')
joblib.dump(le, 'label_encoder.joblib')
logging.info("SUCCESS: 'emotion_model.h5' and 'label_encoder.joblib' created.")
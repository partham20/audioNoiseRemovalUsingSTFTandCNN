import librosa
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import scipy
import scipy.signal as sps
import glob

from keras.models import Model
from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import tensorflow as tf
import keras

def ExtractSTFT(fname,fftLength,hop_length,frame_length):
  fs, x = wavfile.read(fname)
  x = x.astype('float64')
  noise_stft = librosa.stft(x, n_fft=fftLength,hop_length=hop_length)
  noise_stft_mag ,noisy_stft_phase=librosa.magphase(noise_stft)
  noisy_signal_stft_unscaled_magnitude=librosa.amplitude_to_db(noise_stft_mag, ref=np.max)
  y = (noisy_signal_stft_unscaled_magnitude+80)/80
  return y

def reshape(ystft_x):
  ystft_xx = np.zeros((8,128,128,1))
  for m in range(0,1024,128):
    ystft_xx[round(m/128),:,:,0] = ystft_x[:,m:m+128]
  return ystft_xx

def AssembleSTFT(y_pred):
  for m in range(0,8):
    if(m==0):
      y_single = y_pred[m,:,:,0]
    else:
      y_single = np.concatenate((y_single,y_pred[m,:,:,0]),axis=1)
  y_single = np.delete(y_single, range(1016,1024), axis=1)
  return y_single

def InverseSTFT(y,noisy_stft_phase):
  y = y.astype('float64')
  #inverse Process
  y1 = y*80-80
  y11_mag = librosa.db_to_amplitude(y1)
  y12 = y11_mag*np.cos(noisy_stft_phase) + y11_mag*np.sin(noisy_stft_phase)
  y_out = librosa.istft(y12)
  return y_out

def ConvertAudio():
  file_vox = open('/content/drive/MyDrive/voxceleb1/voxceleb1_file_list.txt', 'r')
  file_urb = open('/content/drive/MyDrive/urbansound8k/urbansound8k_file_list.txt', 'r')
  T = 4
  t = np.arange(0,T,1/16000)
  for m in tqdm(range(0,650),desc="Progress.."):
      vox = file_vox.readline()
      urb = file_urb.readline()
      samplerate, x = wavfile.read(vox.strip())
      number_of_samples = round(len(x) * float(16000) / samplerate)
      x = sps.resample(x, number_of_samples)
      x = x.astype('int16')
      samplerate, n = wavfile.read(urb.strip())
      number_of_samples = round(len(n) * float(16000) / samplerate)
      n = sps.resample(n, number_of_samples)
      n = n.astype('int16')
      if(len(n.shape)>1):
        n = n[:,0]
      if (n.shape[0]>=T*16000 and x.shape[0]>=T*16000):
        n = n[0:T*16000]
        x = x[0:T*16000]
        # print([n.shape[0],x.shape[0]])
        x_n = x/2 + n/2
        x_n = x_n.astype('int16')
        scipy.io.wavfile.write('/content/drive/MyDrive/data/train/noisy/'+str(m)+'.wav', 16000, x_n)
        scipy.io.wavfile.write('/content/drive/MyDrive/data/train/original/'+str(m)+'.wav', 16000, x)
  file_vox.close()
  file_urb.close()

def CNNmodel():
  size_filter_in = 16
  #normal initialization of weights
  kernel_init = 'he_normal'
  #To apply leaky relu after the conv layer
  activation_layer = None
  inputs = keras.Input(shape=(128, 128, 1))
  conv1 = Conv2D(size_filter_in, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(inputs)
  conv1 = LeakyReLU()(conv1)
  conv1 = Conv2D(size_filter_in, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv1)
  conv1 = LeakyReLU()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(size_filter_in*2, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool1)
  conv2 = LeakyReLU()(conv2)
  conv2 = Conv2D(size_filter_in*2, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv2)
  conv2 = LeakyReLU()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(size_filter_in*4, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool2)
  conv3 = LeakyReLU()(conv3)
  conv3 = Conv2D(size_filter_in*4, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv3)
  conv3 = LeakyReLU()(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(size_filter_in*8, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool3)
  conv4 = LeakyReLU()(conv4)
  conv4 = Conv2D(size_filter_in*8, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv4)
  conv4 = LeakyReLU()(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(size_filter_in*16, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool4)
  conv5 = LeakyReLU()(conv5)
  conv5 = Conv2D(size_filter_in*16, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv5)
  conv5 = LeakyReLU()(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(size_filter_in*8, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(drop5))
  up6 = LeakyReLU()(up6)
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(size_filter_in*8, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge6)
  conv6 = LeakyReLU()(conv6)
  conv6 = Conv2D(size_filter_in*8, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv6)
  conv6 = LeakyReLU()(conv6)
  up7 = Conv2D(size_filter_in*4, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv6))
  up7 = LeakyReLU()(up7)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(size_filter_in*4, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge7)
  conv7 = LeakyReLU()(conv7)
  conv7 = Conv2D(size_filter_in*4, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv7)
  conv7 = LeakyReLU()(conv7)
  up8 = Conv2D(size_filter_in*2, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv7))
  up8 = LeakyReLU()(up8)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(size_filter_in*2, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge8)
  conv8 = LeakyReLU()(conv8)
  conv8 = Conv2D(size_filter_in*2, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv8)
  conv8 = LeakyReLU()(conv8)

  up9 = Conv2D(size_filter_in, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv8))
  up9 = LeakyReLU()(up9)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(size_filter_in, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge9)
  conv9 = LeakyReLU()(conv9)
  conv9 = Conv2D(size_filter_in, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv9)
  conv9 = LeakyReLU()(conv9)
  conv9 = Conv2D(2, 3, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv9)
  conv9 = LeakyReLU()(conv9)
  conv10 = Conv2D(1, 1, activation = 'tanh')(conv9)

  model = Model(inputs,conv10)
  model.compile(optimizer = 'adam', loss = tf.keras.losses.Huber(), metrics = ['mae'])
  model.summary()

  return model

def WriteVoxFiles():
  f = open("/content/drive/MyDrive/voxceleb1/voxceleb1_file_list.txt", "w")
  fol = glob.glob("/content/drive/MyDrive/voxceleb1/vox1_indian/content/vox_indian/*")
  for m in fol:
    fol_1 = glob.glob(m+'/*')
    for n in fol_1:
      fol_2 = glob.glob(n+'/*.wav')
      for k in fol_2:
        f.write(k+'\n')
  f.close()

def WriteUrbFiles():
  f = open("/content/drive/MyDrive/urbansound8k/urbansound8k_file_list.txt", "w")
  fol = glob.glob("/content/drive/MyDrive/urbansound8k/*")
  for m in range(0,len(fol)):
    fol_1 = glob.glob(fol[m]+'/*.wav')
    for n in fol_1:
      f.write(n+'\n')
      print(n)
  f.close()

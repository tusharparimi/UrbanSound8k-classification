import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
#import tensorflow_io as tfio
from audio_utils import AudioUtil
import pandas as pd

class denseNetAudioTransform():  

    def __init__(self,df,data_path):
        self.df=df
        self.data_path=str(data_path)
        self.duration=4000
        self.sr=44100
        self.channel=2
        self.shift_pct=0.4

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        audio_file=self.data_path+self.df.loc[idx,'relative_path']
        class_id=self.df.loc[idx,'classID']

        aud=AudioUtil.open_wav(audio_file)  
        #some sig are of shape (192000,) when they are supposed to have channels like (1,192000) or (2,192000)
        #-----------------------------------------
        sig,sr=aud
        #print(aud)
        #print('OG shape: ',sig.shape)
        if sig.shape[0]!=1 and sig.shape[0]!=2:
            sig=np.expand_dims(sig, axis=0)
        aud=sig,sr
        #-----------------------------------------
        reaud=AudioUtil.resample(aud,self.sr)
        rechan=AudioUtil.rechannel(reaud,self.channel)
        dur_aud=AudioUtil.pad_trunc(rechan,self.duration)
        shift_aud=AudioUtil.time_shift(dur_aud,self.shift_pct)
        sgrams=AudioUtil.denseNet_spectro_gram(shift_aud, n_mels=128, win_hop_pairs=[(25,10),(50,25),(100,50)])
        #print("new_sgrams :")
        #print(sgrams.shape)
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(sgrams[0], x_axis='time',
        #                     y_axis='mel', sr=44100,
        #                     ax=ax)
        # fig.colorbar(img, ax=ax, format='%+2.0f dB')
        # ax.set(title='Mel-frequency spectrogram')
        # plt.show()
        
        
        aug_sgram=AudioUtil.spectro_augment(sgrams, max_mask_pct=0.1,n_freq_masks=1,n_time_masks=1)
        class_id=to_categorical(class_id, num_classes=10)

        return aug_sgram, class_id
    

if __name__ == "__main__":
        
    df=pd.read_csv("C:\\Users\\tusha\\OneDrive\\Documents\\Projects\\sound-classification\\relativepaths.csv")
    atrans=denseNetAudioTransform(df=df, data_path="C:\\Users\\tusha\\Downloads\\UrbanSound8K\\audio")
    print(atrans.df.head())
    aug_sgram, class_id=atrans[1]
    print(aug_sgram.shape)
    print(type(aug_sgram))
    print(aug_sgram.dtype)
    print(type(class_id))
    print(class_id.shape)
    aug_sgram, class_id=atrans[1]
    print(aug_sgram.shape)
    print(class_id)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(aug_sgram[0], x_axis='time',
                         y_axis='mel', sr=44100,
                         ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(aug_sgram[1], x_axis='time',
                         y_axis='mel', sr=44100,
                         ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(aug_sgram[1], x_axis='time',
                         y_axis='mel', sr=44100,
                         ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()





    


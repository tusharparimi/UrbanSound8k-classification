import librosa
import numpy as np
import random
import tensorflow as tf
#import tensorflow_io as tfio

class AudioUtil():

    @staticmethod
    def open_wav(path):
        sig, sr=librosa.load(path, sr=None, mono=False)
        return (sig, sr)
    
    @staticmethod
    def rechannel(aud, new_channels_num):
        sig,sr=aud
        if sig.shape[0]==new_channels_num:
            return aud
        elif new_channels_num==1:
            resig=sig[:1,:]
        else:
            resig=np.append(sig,sig,axis=0)
        return (resig, sr)
    
    @staticmethod
    def resample(aud,new_sr):
        sig,sr=aud
        #print(sig.shape)
        if sr==new_sr:
            return aud
        num_channels=sig.shape[0]
        resig=librosa.resample(sig[:1,:],orig_sr=sr,target_sr=new_sr)
        if num_channels>1:
            resig2=librosa.resample(sig[1:,:],orig_sr=sr,target_sr=new_sr)
            resig=np.append(resig,resig2,axis=0)
        return (resig, new_sr)

    @staticmethod
    def pad_trunc(aud,max_ms):
        sig,sr=aud
        #print(sig.shape)
        num_channels, sig_len=sig.shape
        max_len=(sr//1000)*max_ms
        if sig_len>max_len:
            resig=sig[:,:max_len]
        elif sig_len<max_len:
            pad_begin_len=random.randint(0,max_len-sig_len)
            pad_end_len=max_len-sig_len-pad_begin_len
            pad_begin=np.zeros((num_channels,pad_begin_len))
            pad_end=np.zeros((num_channels,pad_end_len))
            resig=np.append(pad_begin,sig,axis=1)
            resig=np.append(resig,pad_end,axis=1)
        return (resig,sr)
    
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr=aud
        _,sig_len=sig.shape
        shift_amt=int(random.random()*shift_limit*sig_len)
        return (np.roll(sig,shift_amt),sr)
    
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr=aud
        #print("yooooooo :", sig[0].shape)
        top_db=80
        mel_signal = librosa.feature.melspectrogram(y=sig[0], sr=sr, hop_length=hop_len, n_fft=n_fft, n_mels=n_mels)
        spectrogram = np.abs(mel_signal)
        spec_db = librosa.power_to_db(spectrogram, top_db=top_db)
        return spec_db
    
    @staticmethod
    def denseNet_spectro_gram(aud, n_mels=128, win_hop_pairs=[(25,10),(50,25),(100,50)]):
        sig,sr=aud
        top_db=80
        specs=np.zeros((1,128,400))
        for each in win_hop_pairs:
            n_fft=int((each[0]*sr)/1000)
            hop_len=int((each[1]*sr)/1000)
            #print(n_fft, hop_len)
            mel_signal = librosa.feature.melspectrogram(y=sig[0], sr=sr, hop_length=hop_len, n_fft=n_fft, n_mels=n_mels)
            spectrogram = np.abs(mel_signal)
            spec_db = librosa.power_to_db(spectrogram, top_db=top_db)
            if spec_db.shape[1]<400:
                new_spec_db=np.full((128,400),spec_db.mean())
                new_spec_db[0:spec_db.shape[0],0:spec_db.shape[1]]=spec_db
                spec_db=new_spec_db
            spec_db=np.expand_dims(spec_db,axis=0)
            specs=np.concatenate((specs,spec_db),axis=0)
        return specs[1:]

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        batch_size, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = np.copy(spec)

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            mask_width = np.random.randint(0, freq_mask_param)
            start = np.random.randint(0, n_mels - mask_width)
            aug_spec[:, start:start + mask_width, :] = mask_value

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            mask_width = np.random.randint(0, time_mask_param)
            start = np.random.randint(0, n_steps - mask_width)
            aug_spec[:, :, start:start + mask_width] = mask_value

        return aug_spec
    
    





    



    

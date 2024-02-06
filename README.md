# UrbanSound8k-classification
- In this project the goals is to classify different environmental sounds present in the famous UrbanSound8k dataset like children playing, dog_bark, car_horn etc.This is a audio or sound classification problem. 
- The idea is to use spectgrams inputs to convert our our problem into a image classification problem which are well known to be classified by using the CNNs (Convolutional Neural Networks). 
- Transfer learning was also used to improve the results obtained on a custom base CNN. 
- A Densenet121 pre-trained on imagenet was used to apply transfer learning to improve the performance metrics.

### Results
All result are based on K-fold validation based on the fold provided the dataset owners as performance on these sets are considered a standard.

- Custom CNN

<img width="417" alt="image" src="https://github.com/tusharparimi/UrbanSound8k-classification/assets/93556280/6872dc3a-3284-4e3b-a784-cdcfb36aaf4a">

- Dense121 (pre-trained on Imagenet dataset)

<img width="416" alt="image" src="https://github.com/tusharparimi/UrbanSound8k-classification/assets/93556280/0931fd07-2458-4289-a2c8-b166985e1a53">


### Audio data processing...
- The processing steps are to use the .wav audio files provided by the UrbanSound8k dataset reading them by folds using the tensorflow dataloaders.
- Using custom processing techniques to create the spectograms from the audio files.
- Data augmentation is done to prevent the model from overfitting the training data.
- A bunch of data loading optimization is done to have all the current batch of data in memory for training and pre fetch the next batch to reduce data fetching overhead time.
- In audio processing particular techniques like rechannel, resampling, padding, time-shifting are used to process the data.
- Mel scale (for frequency) and Decibel scale (for ) are used for making the spectrograms


### What are spectograms
- Represntation of audio signals in terms of frequencies as a heat map to see which frequencies were active in a particular time of the audio signal.

<img width="439" alt="image" src="https://github.com/tusharparimi/UrbanSound8k-classification/assets/93556280/bf85688b-d2f5-4079-aa09-83afa8a6fc09">

- Why use Mel and dB scales. Because we humans are more sensitive to differences in lower frequecies than higher frequencies, aand similarly in differentiating amplitude as well.
- For instance, if you listened to different pairs of sound as follows:
  100Hz and 200Hz,
  1000Hz and 1100Hz,
  10000Hz and 10100 Hz
- So you should be distinguish 100Hz from 200Hz more cllearly than 10000Hz from 10100 Hz
- Try it really fun!!! :ok_hand:

### TODO:
- Improve the performance of models by playing with some different pre-trained models
- Refactor and tidy up the code files :unamused:

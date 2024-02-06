import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from audio_trans import AudioTransform
from audio_classifier import AudioClassifier
from audio_utils import AudioUtil
from numpy import savez_compressed
from numpy import load

#read path csv
df=pd.read_csv("C:\\Users\\tusha\\OneDrive\\Documents\\Projects\\sound-classification\\relativepaths.csv")
#df=df.loc[~df['relative_path'].isin(['/fold1/150341-3-1-0.wav'])]

X=df['relative_path']
y=df["classID"]
#split train\test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,test_size=0.2, random_state=1)
#split train\val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True,test_size=0.2, random_state=1)

atrans=AudioTransform(df=df, data_path="C:\\Users\\tusha\\Downloads\\UrbanSound8K\\audio")

def np_array_file_gen(input_set, set_name):
    #generates and saves sgram numpy arrays of provided data sets 
    class_labels=[]
    sgrams=np.zeros((1,64,688,2))
    just=0
    for idx in input_set.index:
        print(just) 
        aug_sgram, class_id = atrans[idx]
        class_labels.append(class_id)
        aug_sgram=np.moveaxis(aug_sgram,0,-1)
        aug_sgram=np.expand_dims(aug_sgram,axis=0)
        sgrams=np.concatenate((sgrams,aug_sgram),axis=0)
        just=just+1
    
    sgrams=sgrams[1:]
    class_labels=np.array(class_labels,ndmin=1,dtype=np.dtype(np.float64))

    savez_compressed(file="C:\\Users\\tusha\\OneDrive\\Desktop"+set_name+'.npz',x=sgrams, y=class_labels)

np_array_file_gen(X_train, "X_train")




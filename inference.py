import pandas as pd
from scipy.signal import butter,filtfilt,find_peaks, argrelmin , argrelmax, peak_prominences,peak_widths
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from tensorflow import keras
from constant import ARTIFACT, NYQ, CUTOFF , ORDER, FS

model = keras.models.load_model(ARTIFACT)

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / NYQ
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# @dataclass
# class Model:
#     model = keras.models.load_model(ARTIFACT)

class Prediction:
    def __init__(self,data:pd.DataFrame, no_of_cols:int=40)->None:
        self.no_of_cols:int = no_of_cols
        self.data:pd.DataFrame = data
        self.threshold:int = None
        self.low_pass:np.ndarray = None
        self.peaks:np.ndarray = None
        self.minima:np.ndarray =  None
        self.dataframe:pd.DataFrame = None
        self.probabilities:np.ndarray = None
    
    @staticmethod
    def create_dataframe(no_of_cols)->pd.DataFrame:
        features = []
        for fe in range(0,no_of_cols):
            features.append(f"feature_{fe}")
        return pd.DataFrame(columns=features)
    
    @staticmethod
    def find_min(minima:np.ndarray,pe:int)->Tuple[int, int]:
        for i in range(len(minima)-1):
            if minima[i] < pe and minima[i+1] > pe:
                return [minima[i], minima[i+1]]
        return [-1,-1]
    
    def preprocess(self)->None:
        self.low_pass = butter_lowpass_filter(self.data["x"], CUTOFF, FS, ORDER)
        self.threshold = np.percentile(self.data["x"],70)
        self.peaks, _ = find_peaks(self.low_pass, height = self.threshold)
        self.minima = argrelmin(self.low_pass)[0]
        self.dataframe = Prediction.create_dataframe(self.no_of_cols)
        
    
    def prediction(self,model=model, draw:bool=False):
        self.preprocess()
        for i_,pe in enumerate(self.peaks):
            ans:Tuple[int, int] = Prediction.find_min(self.minima,pe)
            if ans == [-1,-1]:
                if i_ == 0:
                    ans:Tuple[int, int] = [0,self.minima[0]]
                elif i_ == len(self.peaks)-1:
                    ans:Tuple[int, int] = [self.minima[-1],-1]        
            arr:np.ndarray = np.array(list(self.data["y"][ans[0]:ans[1]]))
            if len(arr)<self.no_of_cols:
                arr = np.pad(arr, (0, self.no_of_cols-len(arr)), mode="constant")
            else:
                arr = arr[:self.no_of_cols]
            if not np.all(arr==0):
                self.dataframe.loc[len(self.dataframe)] = arr
                
        self.predict(model)
        if draw:
            self.draw_steps()
        
    def predict(self,model)->None:
        self.probabilities = model.predict(self.dataframe)
        self.dataframe["prediction"] = np.argmax(self.probabilities, axis=-1)
#         self.dataframe["prediction"] = model.predict(self.dataframe)
        self.dataframe["prediction"].replace(0,"left",inplace=True)
        self.dataframe["prediction"].replace(1,"right",inplace=True)
    
    def draw_steps(self)->None:
        print(f"Number of peaks in the given data: {len(self.peaks)}")
        plt.figure(figsize=(20, 4))
        plt.plot(self.low_pass)
        plt.plot(self.data["y"])
        plt.plot(self.peaks,self.low_pass[self.peaks], "X")
        plt.plot(self.minima,self.low_pass[self.minima], "X", color="black")
        for p,s in zip(self.peaks,self.dataframe["prediction"].values):
            if s == "left":
                plt.text(p-10,self.low_pass[p], s, fontsize = 12, color="black")
            else:
                plt.text(p-10,self.low_pass[p], s, fontsize = 12, color="red")
        plt.show()
        
        data1 = self.dataframe[self.dataframe["prediction"]=="left"]
        data2 = self.dataframe[self.dataframe["prediction"]=="right"]
        plt.figure(figsize=(20, 3))
        plt.subplot(1, 2, 1)
        plt.plot(data1.iloc[:,:-1].transpose())

        plt.subplot(1, 2, 2)
        plt.plot(data2.iloc[:,:-1].transpose())
        plt.show()    


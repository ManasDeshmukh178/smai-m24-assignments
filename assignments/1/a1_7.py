import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('dataset.csv')
numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 
                      'key', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo']
data = df[numerical_features].to_numpy()
# array([[ 50, 210000, 0.8],
#        [ 65, 180000, 0.6],
#        [ 40, 240000, 0.7]])
print(data)  #printing list of each table entry in dataset.csv file

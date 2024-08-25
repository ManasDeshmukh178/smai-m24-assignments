import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('dataset.csv')
numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 
                      'key', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo']  #enlisting features
data = df[numerical_features].to_numpy()
plt.figure(figsize=(20, 15))

for i, feature in enumerate(numerical_features):   # plotting for all numjerical features
    plt.subplot(4, 3, i+1)
    values = data[:, i]
    counts, bins = np.histogram(values, bins=30)
    plt.hist(values, bins=30, alpha=0.75, color='green', edgecolor='blue')#plotting for each quantity
    plt.title(f' {feature}')
    plt.ylabel('Frequency')
plt.savefig('q1_res1.jpg')  #saving plot
plt.tight_layout()
plt.show() #showing plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset.csv')

# Define numerical features
numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 
                      'key', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo']

# Extract numerical data
data = df[numerical_features].to_numpy()

# Compute correlation matrix
corr_matrix = np.corrcoef(data, rowvar=False)

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Correlation coefficient')
plt.xticks(range(len(numerical_features)), numerical_features, rotation=90)
plt.yticks(range(len(numerical_features)), numerical_features)
plt.title('Correlation Matrix')

# Save the plot as an image
plt.savefig('q1_res2.jpg')

# Display the plot
plt.show()

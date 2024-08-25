import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset.csv')

# List of numerical features
numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 
                      'key', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo']

# Check for missing values and basic statistics
print(df[numerical_features].describe())
print(df[numerical_features].isnull().sum())

# Plot distributions of each numerical feature and close figures after plotting
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    plt.hist(df[feature].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.close()

# Plot correlations with the target variable 'track_genre'
target = 'key'

# Ensure the target column exists
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in the dataset")

# Encode the target variable using one-hot encoding
encoded_genres = pd.get_dummies(df[target])

# Compute and plot correlations for each genre
for genre in encoded_genres.columns:
    plt.figure(figsize=(12, 8))
    correlation = df[numerical_features].corrwith(encoded_genres[genre])
    plt.bar(correlation.index, correlation)
    plt.title(f'Correlation of Features with {genre}')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    plt.close()

# Hierarchy of features based on correlation with target variable
mean_correlations = df[numerical_features].corrwith(encoded_genres.mean())
sorted_features = mean_correlations.abs().sort_values(ascending=False)   #sorting values
print("Feature Importance based on Correlation with Target Variable:")
print(sorted_features)

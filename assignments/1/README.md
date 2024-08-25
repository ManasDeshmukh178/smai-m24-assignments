# Assignment 1 Report:-
Explanation of KNN Implementation and Analysis
1. Exploratory Data Analysis (EDA)

Feature Distributions:

    Histograms and Box Plots: These plots help visualize the distribution and spread of individual features. For instance, histograms can show if features like danceability are uniformly distributed or skewed. Box plots can reveal outliers, such as unusually high or low loudness values.
    Scatter Plots: These are useful for examining the relationships between features and the target variable (genre). For example, plotting energy versus danceability might show a correlation that could be relevant for predicting genre.

Feature Correlation with Target:

    Correlation Matrix: Calculating correlations between features and the target variable can help identify which features are most predictive. High correlations suggest that a feature has a strong relationship with the genre and could be crucial for the model.
    Pair Plots: Visualizing combinations of features can show how different features interact and their collective impact on the target variable.

Feature Importance Hierarchy:

    After performing EDA, you should rank features based on their impact on the target variable. Features with strong correlations or significant patterns are considered more important. This hierarchy helps in feature selection and improves model performance by focusing on the most relevant features.

2. KNN Implementation

Class Design:

    Initialization: The KNN class is initialized with parameters for the number of neighbors (k) and the distance metric (e.g., Euclidean or Manhattan distance). This design allows flexibility in tuning the model.
    Fit Method: The fit method stores training data, which is used for making predictions.
    Predict Method: The predict method uses the trained data to classify new instances. It involves computing distances between the new instance and all training instances, selecting the nearest neighbors, and predicting the majority class.
    Distance Computation: Depending on the chosen distance metric, distances between data points are calculated differently. The Euclidean distance measures the straight-line distance between points, while Manhattan distance sums the absolute differences along each dimension.

Metrics Calculation:

    Accuracy: Measures the proportion of correct predictions. It’s straightforward and helps in understanding the overall performance.
    Precision and Recall: These metrics evaluate the performance more granularly. Precision measures the proportion of true positive predictions among all positive predictions, while recall measures the proportion of true positives among all actual positives. For multi-class classification, these metrics can be averaged (macro or micro) to get a general sense of performance across different classes.

3. Hyperparameter Tuning

Finding Optimal Parameters:

    Validation Accuracy: The goal is to find the k and distance metric that yield the highest accuracy on the validation set. This involves training and evaluating the model with various combinations of these parameters.
    Top 10 Pairs: By ranking {k, distance metric} pairs based on validation accuracy, you can identify which configurations work best.

Plotting and Analysis:

    K vs. Accuracy: Plotting how accuracy changes with different values of k can help in understanding the impact of k on model performance. It’s important to choose a value of k that balances between underfitting and overfitting.

4. Optimization

Vectorization:

    Execution Time Improvement: Vectorization replaces explicit loops with array operations, improving efficiency. This is crucial for handling large datasets.

Inference Time Comparison:

    Initial vs. Optimized Models: Compare the inference time of the initial KNN implementation, the best KNN model, and optimized versions. This helps in understanding the performance gains achieved through optimization.

Dataset Size Impact:

    Inference Time vs. Dataset Size: Plotting inference time against dataset size for different models can reveal how well the models scale with data. Observations can guide further optimizations and adjustments in model complexity.

5. Applying the Best Model

Second Dataset:

    Evaluation on New Data: Apply the best {k, distance metric} pair to the new dataset split into training, validation, and test sets. Analyze performance to ensure accuracy falls within the desired range (0.25 to 0.32).
    Observations and Learnings: Document insights gained from applying the model, including how well it generalizes to new data and any differences observed compared to the initial dataset. This analysis helps in understanding the model’s robustness and suitability for genre prediction



    Linear Regression Implementation and Analysis
1. Simple Regression

Implementation Details:

    Data Handling: For linreg.csv, after shuffling the data, we split it into training, validation, and test sets in an 80:10:10 ratio. This ensures a robust evaluation of the model's performance.
    Model Fitting: Implementing linear regression for a degree 1 polynomial involves finding the best fit line y=β1x+β0y=β1​x+β0​. This requires solving the normal equations derived from the least squares method. Specifically:
    w=(XTX)−1XTy
    w=(XTX)−1XTy Here, ww contains the coefficients β0β0​ and β1β1​, and XX includes a column of ones for the intercept term.
    Metrics Calculation: For evaluating the model, calculate Mean Squared Error (MSE), standard deviation, and variance for both the training and test sets. These metrics provide insights into the model's performance and its generalization capability.

Degree > 1 - Polynomial Regression:

    Polynomial Features: Extend the linear model to polynomial regression by transforming the features. For a degree kk, each feature is raised to the power of kk and combined with other polynomial terms.
    Model Fitting: Use matrix operations to solve for polynomial coefficients:
    w=(XTX)−1XTy
    w=(XTX)−1XTy where XX now includes polynomial terms up to degree kk.
    Model Evaluation: Assess the polynomial models' performance using MSE, standard deviation, and variance for different degrees. The goal is to identify the degree kk that minimizes test set error while avoiding overfitting.

Animation:

    Visualization: Create a GIF that shows how the polynomial fit evolves over iterations. Include plots of the original data, the fitted polynomial, and metrics such as MSE, standard deviation, and variance. This visualization helps in understanding the fitting process and convergence behavior.

2. Regularization

Implementation Details:

    Regularization Techniques: Integrate L1 and L2 regularization into the regression model:
        L1 Regularization (Lasso): Adds a penalty proportional to the absolute value of the coefficients:
        Cost Function=MSE+λ∑j∣βj∣
        Cost Function=MSE+λj∑​∣βj​∣ L1 regularization promotes sparsity, which can lead to some coefficients being zero.
        L2 Regularization (Ridge): Adds a penalty proportional to the square of the coefficients:
        Cost Function=MSE+λ∑jβj2
        Cost Function=MSE+λj∑​βj2​ L2 regularization smooths the coefficients and can handle multicollinearity better than L1.
    Polynomial Features and Regularization: When applying regularization, fit higher-degree polynomials to the data and incorporate regularization into the cost function to prevent overfitting.
    Model Evaluation: Report MSE, standard deviation, and variance for different polynomial degrees with and without regularization. Visualize the fitted curves to understand how regularization affects model complexity and performance.

Observations:

    Overfitting: Higher-degree polynomials without regularization tend to fit noise in the data, leading to poor generalization. Regularization helps mitigate this by constraining the model complexity.
    Effectiveness of Regularization: L1 regularization can produce sparser models with fewer non-zero coefficients, while L2 regularization tends to produce models with smaller, more evenly distributed coefficients.
    Performance Comparison: By comparing regularized and non-regularized models, you can assess how well each approach handles the trade-off between model complexity and accuracy

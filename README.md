# CSIS3360_DAProject

Melbourne Real Estate Price Prediction

Objective

To determine the most effective machine learning model for predicting housing price categories in Melbourne, Australia. Rather than forecasting exact price values, we approached the problem as a classification task, categorizing properties into Low, Medium, or High price tiers based on their value.

Dataset

Source: Melbourne Housing Market dataset (CSV)

Records after cleaning: ~13,580

Target Variable: PriceCategory (created using quantile thresholds)

Feature Types

Numerical Features: Rooms, Distance, Landsize, BuildingArea, YearBuilt, Bathroom, Bedroom2, Car, Propertycount

Categorical Features: Suburb, Type, Method, SellerG, CouncilArea, Regionname

Target Transformation

The continuous price column was transformed into a categorical variable:

Bottom 33% → Low

Middle 34% → Medium

Top 33% → High

Data Preparation

Dropped irrelevant columns: Address, Postcode, raw Date

Extracted new features: Year, Month, Day, AreaRatio, Age, RoomsPerBedroom

Filled missing values (median for numeric, "Unknown" for categorical)

Capped outliers at 99th percentile

One-hot and label encoded categorical variables

Handling Imbalance

Applied Stratified Train-Test Split to maintain class distribution

Used SMOTE (Synthetic Minority Oversampling Technique) on the training data to address class imbalance

Model Comparison (Classification)

Three machine learning models were trained and evaluated using accuracy and confusion matrices:

1. Random Forest (with Optuna tuning)

Ensemble of decision trees

Accuracy: 81.3%

2. LightGBM (with Optuna tuning)

Fast and efficient gradient boosting model

Accuracy: 81.5%

3. SVM (Support Vector Machine)

Effective for high-dimensional spaces, but lower performance here

Accuracy: 75.9%

Visualization

Side-by-side confusion matrices for each model

Accuracy comparison bar chart

Conclusion

LightGBM delivered the highest accuracy among all models. Combining feature engineering, SMOTE, and hyperparameter tuning significantly improved performance. Classification turned out to be more effective than regression for this dataset.
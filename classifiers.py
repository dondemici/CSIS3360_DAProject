# ------------------------------
# 0. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for model building and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Imbalanced-learn for oversampling and pipeline (ensures oversampling is applied within CV only)
#%pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
#!pip install optuna
import optuna


# (Optional) If running in Google Colab, mount your Google Drive:
#%pip install google-colab
#try:
#    try:
#        from google.colab import drive
#    except ImportError:
#        drive = None
#    drive.mount('/content/drive')
#except ImportError:
#    print("Not running in Google Colab. Skipping Google Drive mounting.")

# ------------------------------
# 1. Load and Preprocess Data
# ------------------------------
# Adjust file_path to your data location in your Google Drive
file_path = r"C:\Users\PCAdmin\Documents\Projects\MELRE_PricePrediction\Melbourne_RealEstate.csv"
#file_path = 'drive/MyDrive/Colab Notebooks/Melbourne_RealEstate.csv'
df = pd.read_csv(file_path)

# Drop columns that aren’t needed
if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

# ------------------------------
# 2. Handle Missing Values & Outliers
# ------------------------------

# Define column types
numeric_cols = ['Rooms', 'Price', 'Bedroom2', 'Bathroom', 'Car', 'Distance',
                'Propertycount', 'Landsize', 'BuildingArea']
categorical_cols = ['Method', 'Type', 'SellerG', 'Regionname', 'CouncilArea']
date_cols = ['Date']

# Fill missing numeric values with the median
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with "Unknown"
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

# Process date columns: convert to datetime, fill missing values using forward fill
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        df[col].fillna(method='ffill', inplace=True)

# Cap outliers in selected columns at the 99th percentile
outlier_cols = ['Price', 'Landsize', 'BuildingArea']
for col in outlier_cols:
    if col in df.columns:
        upper_limit = df[col].quantile(0.99)
        df.loc[df[col] > upper_limit, col] = upper_limit

# ------------------------------
# 3. Enhanced Feature Engineering
# ------------------------------
# Create date-derived features (Year, Month, Day)
if 'Date' in df.columns:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

# Example: Create an interaction feature using BuildingArea and Landsize (avoid division by zero)
df['AreaRatio'] = np.where(df['Landsize'] > 0, df['BuildingArea'] / df['Landsize'], 0)

# (Optional) If a 'YearBuilt' column exists, compute property Age at sale using the 'Year' feature.
if 'YearBuilt' in df.columns:
    df['Age'] = df['Year'] - df['YearBuilt']
    df['Age'] = df['Age'].apply(lambda x: x if x > 0 else 0)

# Create additional interaction features: e.g., Rooms per Bedroom
if 'Bedroom2' in df.columns:
    df['RoomsPerBedroom'] = np.where(df['Bedroom2'] > 0, df['Rooms'] / df['Bedroom2'], df['Rooms'])

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------
# Use get_dummies for categorical variables (drop_first to avoid collinearity)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# For high-cardinality features such as Suburb use Label Encoding if available
if 'Suburb' in df.columns:
    le = LabelEncoder()
    df['Suburb'] = le.fit_transform(df['Suburb'])

# ------------------------------
# 5. Create Classification Target (Price Category)
# ------------------------------
# Use quantile thresholds to create a categorical variable:
q1 = df['Price'].quantile(0.33)
q2 = df['Price'].quantile(0.67)

def classify_price(price):
    if price <= q1:
        return 0  # Low price
    elif price <= q2:
        return 1  # Medium price
    else:
        return 2  # High price

df['PriceCategory'] = df['Price'].apply(classify_price)
# Remove original Price from features
df.drop('Price', axis=1, inplace=True)

# ------------------------------
# 6. Define Features and Target
# ------------------------------
target = 'PriceCategory'
X = df.drop(columns=[target])
y = df[target]

# Fill any remaining missing values (if any) in features with 0
X.fillna(0, inplace=True)

# ------------------------------
# 7. Split Data into Train/Test sets
# ------------------------------
# Stratify split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# ------------------------------
# 8. Build an Imbalanced-Learn Pipeline
# ------------------------------
# The pipeline includes:
#  - StandardScaler for numerical feature scaling
#  - SMOTE for oversampling the minority classes (applied after scaling)
#  - SelectKBest for feature selection (we let k be optimized)
#  - RandomForestClassifier for prediction

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
])

# ------------------------------
# 9. Hyperparameter Tuning with Optuna
# ------------------------------
def objective(trial):
    # Hyperparameters for SelectKBest: number of features to select (between 5 and total features)
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    # Hyperparameters for Random Forest
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Set parameters for the pipeline steps
    pipeline.set_params(select__k=k,
                          rf__n_estimators=n_estimators,
                          rf__max_depth=max_depth,
                          rf__min_samples_split=min_samples_split,
                          rf__min_samples_leaf=min_samples_leaf,
                          rf__criterion=criterion)

    # Use Stratified K-Fold Cross-Validation for robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

    # Return the average accuracy across folds
    return scores.mean()

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Increase number of trials if computationally feasible

print("Best trial:")
trial = study.best_trial
print("  Accuracy: {:.4f}".format(trial.value))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# ------------------------------
# 10. Train Final Model with Best Hyperparameters & Evaluate
# ------------------------------
# Set the pipeline with the best hyperparameters found from Optuna
pipeline.set_params(select__k=trial.params['select_k'],
                    rf__n_estimators=trial.params['n_estimators'],
                    rf__max_depth=trial.params['max_depth'],
                    rf__min_samples_split=trial.params['min_samples_split'],
                    rf__min_samples_leaf=trial.params['min_samples_leaf'],
                    rf__criterion=trial.params['criterion'])

# Fit the pipeline on the full training set
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy: {:.4f}".format(test_accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Display train and test set sizes
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ------------------------------
# 10b. Add optuna
# ------------------------------

from lightgbm import LGBMClassifier

def lgbm_objective(trial):
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    pipeline_lgbm_opt = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
        ('lgbm', LGBMClassifier(**params))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_lgbm_opt, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Run the tuning
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(lgbm_objective, n_trials=30)

# Store best params
lgbm_best_params = lgbm_study.best_params
print("\n✅ Tuned LightGBM Accuracy: {:.4f}".format(lgbm_study.best_value))
print("Best LightGBM Parameters:")
for k, v in lgbm_best_params.items():
    print(f"  {k}: {v}")


# ------------------------------
# 11. Train and Evaluate LightGBM Classifier
# ------------------------------
from lightgbm import LGBMClassifier

# Reuse the same pipeline setup, replacing only the classifier
#pipeline_lgbm = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # use same k
#    ('lgbm', LGBMClassifier(random_state=42))
#])

pipeline_lgbm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=lgbm_best_params['select_k'])),
    ('lgbm', LGBMClassifier(
        n_estimators=lgbm_best_params['n_estimators'],
        learning_rate=lgbm_best_params['learning_rate'],
        num_leaves=lgbm_best_params['num_leaves'],
        max_depth=lgbm_best_params['max_depth'],
        min_child_samples=lgbm_best_params['min_child_samples'],
        subsample=lgbm_best_params['subsample'],
        colsample_bytree=lgbm_best_params['colsample_bytree'],
        random_state=42
    ))
])

# Fit LGBM pipeline
pipeline_lgbm.fit(X_train, y_train)

# Predict
y_pred_lgbm = pipeline_lgbm.predict(X_test)

# Evaluate
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
print("\nLightGBM Test Accuracy: {:.4f}".format(lgbm_accuracy))
print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm))

# Confusion Matrix for LightGBM
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)


# ------------------------------
# 12. Plot Confusion Matrices Side-by-Side
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()


# ------------------------------
# 14. SVM
# ------------------------------

from sklearn.svm import SVC

# Build the same pipeline with SVM
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # reuse k
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# Fit pipeline
pipeline_svm.fit(X_train, y_train)

# Predict
y_pred_svm = pipeline_svm.predict(X_test)

# Evaluate
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Test Accuracy: {:.4f}".format(svm_accuracy))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)


#from catboost import CatBoostClassifier

#pipeline_cat = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),
#    ('catboost', CatBoostClassifier(
#        verbose=0,
#        iterations=300,
#        learning_rate=0.1,
#        depth=6,
#        random_state=42
#    ))
#])

#pipeline_cat.fit(X_train, y_train)
#y_pred_cat = pipeline_cat.predict(X_test)
#cat_accuracy = accuracy_score(y_test, y_pred_cat)
#print("\nCatBoost Test Accuracy: {:.4f}".format(cat_accuracy))
#print("\nCatBoost Classification Report:\n", classification_report(y_test, y_pred_cat))

# Confusion matrix
#conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[2])
axes[2].set_title(f"CatBoost (Acc: {svm_accuracy:.2f})")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# ------------------------------
# 15. Accuracy Comparison Bar Chart
# ------------------------------
#model_names = ['Random Forest', 'LightGBM', 'SVM']
#accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

model_names = ['Random Forest', 'LightGBM', 'SVM']
accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='Set2')
plt.ylim(0.70, 0.85)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xlabel("Model")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.3f}", ha='center', va='bottom', fontsize=10)
plt.show()# ------------------------------
# 0. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for model building and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Imbalanced-learn for oversampling and pipeline (ensures oversampling is applied within CV only)
#%pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
#!pip install optuna
import optuna


# (Optional) If running in Google Colab, mount your Google Drive:
#%pip install google-colab
#try:
#    try:
#        from google.colab import drive
#    except ImportError:
#        drive = None
#    drive.mount('/content/drive')
#except ImportError:
#    print("Not running in Google Colab. Skipping Google Drive mounting.")

# ------------------------------
# 1. Load and Preprocess Data
# ------------------------------
# Adjust file_path to your data location in your Google Drive
file_path = r"C:\Users\PCAdmin\Documents\Projects\MELRE_PricePrediction\Melbourne_RealEstate.csv"
#file_path = 'drive/MyDrive/Colab Notebooks/Melbourne_RealEstate.csv'
df = pd.read_csv(file_path)

# Drop columns that aren’t needed
if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

# ------------------------------
# 2. Handle Missing Values & Outliers
# ------------------------------

# Define column types
numeric_cols = ['Rooms', 'Price', 'Bedroom2', 'Bathroom', 'Car', 'Distance',
                'Propertycount', 'Landsize', 'BuildingArea']
categorical_cols = ['Method', 'Type', 'SellerG', 'Regionname', 'CouncilArea']
date_cols = ['Date']

# Fill missing numeric values with the median
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with "Unknown"
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

# Process date columns: convert to datetime, fill missing values using forward fill
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        df[col].fillna(method='ffill', inplace=True)

# Cap outliers in selected columns at the 99th percentile
outlier_cols = ['Price', 'Landsize', 'BuildingArea']
for col in outlier_cols:
    if col in df.columns:
        upper_limit = df[col].quantile(0.99)
        df.loc[df[col] > upper_limit, col] = upper_limit

# ------------------------------
# 3. Enhanced Feature Engineering
# ------------------------------
# Create date-derived features (Year, Month, Day)
if 'Date' in df.columns:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

# Example: Create an interaction feature using BuildingArea and Landsize (avoid division by zero)
df['AreaRatio'] = np.where(df['Landsize'] > 0, df['BuildingArea'] / df['Landsize'], 0)

# (Optional) If a 'YearBuilt' column exists, compute property Age at sale using the 'Year' feature.
if 'YearBuilt' in df.columns:
    df['Age'] = df['Year'] - df['YearBuilt']
    df['Age'] = df['Age'].apply(lambda x: x if x > 0 else 0)

# Create additional interaction features: e.g., Rooms per Bedroom
if 'Bedroom2' in df.columns:
    df['RoomsPerBedroom'] = np.where(df['Bedroom2'] > 0, df['Rooms'] / df['Bedroom2'], df['Rooms'])

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------
# Use get_dummies for categorical variables (drop_first to avoid collinearity)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# For high-cardinality features such as Suburb use Label Encoding if available
if 'Suburb' in df.columns:
    le = LabelEncoder()
    df['Suburb'] = le.fit_transform(df['Suburb'])

# ------------------------------
# 5. Create Classification Target (Price Category)
# ------------------------------
# Use quantile thresholds to create a categorical variable:
q1 = df['Price'].quantile(0.33)
q2 = df['Price'].quantile(0.67)

def classify_price(price):
    if price <= q1:
        return 0  # Low price
    elif price <= q2:
        return 1  # Medium price
    else:
        return 2  # High price

df['PriceCategory'] = df['Price'].apply(classify_price)
# Remove original Price from features
df.drop('Price', axis=1, inplace=True)

# ------------------------------
# 6. Define Features and Target
# ------------------------------
target = 'PriceCategory'
X = df.drop(columns=[target])
y = df[target]

# Fill any remaining missing values (if any) in features with 0
X.fillna(0, inplace=True)

# ------------------------------
# 7. Split Data into Train/Test sets
# ------------------------------
# Stratify split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# ------------------------------
# 8. Build an Imbalanced-Learn Pipeline
# ------------------------------
# The pipeline includes:
#  - StandardScaler for numerical feature scaling
#  - SMOTE for oversampling the minority classes (applied after scaling)
#  - SelectKBest for feature selection (we let k be optimized)
#  - RandomForestClassifier for prediction

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
])

# ------------------------------
# 9. Hyperparameter Tuning with Optuna
# ------------------------------
def objective(trial):
    # Hyperparameters for SelectKBest: number of features to select (between 5 and total features)
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    # Hyperparameters for Random Forest
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Set parameters for the pipeline steps
    pipeline.set_params(select__k=k,
                          rf__n_estimators=n_estimators,
                          rf__max_depth=max_depth,
                          rf__min_samples_split=min_samples_split,
                          rf__min_samples_leaf=min_samples_leaf,
                          rf__criterion=criterion)

    # Use Stratified K-Fold Cross-Validation for robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

    # Return the average accuracy across folds
    return scores.mean()

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Increase number of trials if computationally feasible

print("Best trial:")
trial = study.best_trial
print("  Accuracy: {:.4f}".format(trial.value))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# ------------------------------
# 10. Train Final Model with Best Hyperparameters & Evaluate
# ------------------------------
# Set the pipeline with the best hyperparameters found from Optuna
pipeline.set_params(select__k=trial.params['select_k'],
                    rf__n_estimators=trial.params['n_estimators'],
                    rf__max_depth=trial.params['max_depth'],
                    rf__min_samples_split=trial.params['min_samples_split'],
                    rf__min_samples_leaf=trial.params['min_samples_leaf'],
                    rf__criterion=trial.params['criterion'])

# Fit the pipeline on the full training set
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy: {:.4f}".format(test_accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Display train and test set sizes
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ------------------------------
# 10b. Add optuna
# ------------------------------

from lightgbm import LGBMClassifier

def lgbm_objective(trial):
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    pipeline_lgbm_opt = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
        ('lgbm', LGBMClassifier(**params))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_lgbm_opt, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Run the tuning
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(lgbm_objective, n_trials=30)

# Store best params
lgbm_best_params = lgbm_study.best_params
print("\n✅ Tuned LightGBM Accuracy: {:.4f}".format(lgbm_study.best_value))
print("Best LightGBM Parameters:")
for k, v in lgbm_best_params.items():
    print(f"  {k}: {v}")


# ------------------------------
# 11. Train and Evaluate LightGBM Classifier
# ------------------------------
from lightgbm import LGBMClassifier

# Reuse the same pipeline setup, replacing only the classifier
#pipeline_lgbm = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # use same k
#    ('lgbm', LGBMClassifier(random_state=42))
#])

pipeline_lgbm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=lgbm_best_params['select_k'])),
    ('lgbm', LGBMClassifier(
        n_estimators=lgbm_best_params['n_estimators'],
        learning_rate=lgbm_best_params['learning_rate'],
        num_leaves=lgbm_best_params['num_leaves'],
        max_depth=lgbm_best_params['max_depth'],
        min_child_samples=lgbm_best_params['min_child_samples'],
        subsample=lgbm_best_params['subsample'],
        colsample_bytree=lgbm_best_params['colsample_bytree'],
        random_state=42
    ))
])

# Fit LGBM pipeline
pipeline_lgbm.fit(X_train, y_train)

# Predict
y_pred_lgbm = pipeline_lgbm.predict(X_test)

# Evaluate
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
print("\nLightGBM Test Accuracy: {:.4f}".format(lgbm_accuracy))
print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm))

# Confusion Matrix for LightGBM
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)


# ------------------------------
# 12. Plot Confusion Matrices Side-by-Side
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()


# ------------------------------
# 14. SVM
# ------------------------------

from sklearn.svm import SVC

# Build the same pipeline with SVM
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # reuse k
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# Fit pipeline
pipeline_svm.fit(X_train, y_train)

# Predict
y_pred_svm = pipeline_svm.predict(X_test)

# Evaluate
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Test Accuracy: {:.4f}".format(svm_accuracy))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)


#from catboost import CatBoostClassifier

#pipeline_cat = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),
#    ('catboost', CatBoostClassifier(
#        verbose=0,
#        iterations=300,
#        learning_rate=0.1,
#        depth=6,
#        random_state=42
#    ))
#])

#pipeline_cat.fit(X_train, y_train)
#y_pred_cat = pipeline_cat.predict(X_test)
#cat_accuracy = accuracy_score(y_test, y_pred_cat)
#print("\nCatBoost Test Accuracy: {:.4f}".format(cat_accuracy))
#print("\nCatBoost Classification Report:\n", classification_report(y_test, y_pred_cat))

# Confusion matrix
#conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[2])
axes[2].set_title(f"CatBoost (Acc: {svm_accuracy:.2f})")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# ------------------------------
# 15. Accuracy Comparison Bar Chart
# ------------------------------
#model_names = ['Random Forest', 'LightGBM', 'SVM']
#accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

model_names = ['Random Forest', 'LightGBM', 'SVM']
accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='Set2')
plt.ylim(0.70, 0.85)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xlabel("Model")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.3f}", ha='center', va='bottom', fontsize=10)
plt.show()# ------------------------------
# 0. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for model building and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Imbalanced-learn for oversampling and pipeline (ensures oversampling is applied within CV only)
#%pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
#!pip install optuna
import optuna


# (Optional) If running in Google Colab, mount your Google Drive:
#%pip install google-colab
#try:
#    try:
#        from google.colab import drive
#    except ImportError:
#        drive = None
#    drive.mount('/content/drive')
#except ImportError:
#    print("Not running in Google Colab. Skipping Google Drive mounting.")

# ------------------------------
# 1. Load and Preprocess Data
# ------------------------------
# Adjust file_path to your data location in your Google Drive
file_path = r"C:\Users\PCAdmin\Documents\Projects\MELRE_PricePrediction\Melbourne_RealEstate.csv"
#file_path = 'drive/MyDrive/Colab Notebooks/Melbourne_RealEstate.csv'
df = pd.read_csv(file_path)

# Drop columns that aren’t needed
if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

# ------------------------------
# 2. Handle Missing Values & Outliers
# ------------------------------

# Define column types
numeric_cols = ['Rooms', 'Price', 'Bedroom2', 'Bathroom', 'Car', 'Distance',
                'Propertycount', 'Landsize', 'BuildingArea']
categorical_cols = ['Method', 'Type', 'SellerG', 'Regionname', 'CouncilArea']
date_cols = ['Date']

# Fill missing numeric values with the median
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with "Unknown"
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

# Process date columns: convert to datetime, fill missing values using forward fill
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        df[col].fillna(method='ffill', inplace=True)

# Cap outliers in selected columns at the 99th percentile
outlier_cols = ['Price', 'Landsize', 'BuildingArea']
for col in outlier_cols:
    if col in df.columns:
        upper_limit = df[col].quantile(0.99)
        df.loc[df[col] > upper_limit, col] = upper_limit

# ------------------------------
# 3. Enhanced Feature Engineering
# ------------------------------
# Create date-derived features (Year, Month, Day)
if 'Date' in df.columns:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

# Example: Create an interaction feature using BuildingArea and Landsize (avoid division by zero)
df['AreaRatio'] = np.where(df['Landsize'] > 0, df['BuildingArea'] / df['Landsize'], 0)

# (Optional) If a 'YearBuilt' column exists, compute property Age at sale using the 'Year' feature.
if 'YearBuilt' in df.columns:
    df['Age'] = df['Year'] - df['YearBuilt']
    df['Age'] = df['Age'].apply(lambda x: x if x > 0 else 0)

# Create additional interaction features: e.g., Rooms per Bedroom
if 'Bedroom2' in df.columns:
    df['RoomsPerBedroom'] = np.where(df['Bedroom2'] > 0, df['Rooms'] / df['Bedroom2'], df['Rooms'])

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------
# Use get_dummies for categorical variables (drop_first to avoid collinearity)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# For high-cardinality features such as Suburb use Label Encoding if available
if 'Suburb' in df.columns:
    le = LabelEncoder()
    df['Suburb'] = le.fit_transform(df['Suburb'])

# ------------------------------
# 5. Create Classification Target (Price Category)
# ------------------------------
# Use quantile thresholds to create a categorical variable:
q1 = df['Price'].quantile(0.33)
q2 = df['Price'].quantile(0.67)

def classify_price(price):
    if price <= q1:
        return 0  # Low price
    elif price <= q2:
        return 1  # Medium price
    else:
        return 2  # High price

df['PriceCategory'] = df['Price'].apply(classify_price)
# Remove original Price from features
df.drop('Price', axis=1, inplace=True)

# ------------------------------
# 6. Define Features and Target
# ------------------------------
target = 'PriceCategory'
X = df.drop(columns=[target])
y = df[target]

# Fill any remaining missing values (if any) in features with 0
X.fillna(0, inplace=True)

# ------------------------------
# 7. Split Data into Train/Test sets
# ------------------------------
# Stratify split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# ------------------------------
# 8. Build an Imbalanced-Learn Pipeline
# ------------------------------
# The pipeline includes:
#  - StandardScaler for numerical feature scaling
#  - SMOTE for oversampling the minority classes (applied after scaling)
#  - SelectKBest for feature selection (we let k be optimized)
#  - RandomForestClassifier for prediction

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
])

# ------------------------------
# 9. Hyperparameter Tuning with Optuna
# ------------------------------
def objective(trial):
    # Hyperparameters for SelectKBest: number of features to select (between 5 and total features)
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    # Hyperparameters for Random Forest
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Set parameters for the pipeline steps
    pipeline.set_params(select__k=k,
                          rf__n_estimators=n_estimators,
                          rf__max_depth=max_depth,
                          rf__min_samples_split=min_samples_split,
                          rf__min_samples_leaf=min_samples_leaf,
                          rf__criterion=criterion)

    # Use Stratified K-Fold Cross-Validation for robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

    # Return the average accuracy across folds
    return scores.mean()

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Increase number of trials if computationally feasible

print("Best trial:")
trial = study.best_trial
print("  Accuracy: {:.4f}".format(trial.value))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# ------------------------------
# 10. Train Final Model with Best Hyperparameters & Evaluate
# ------------------------------
# Set the pipeline with the best hyperparameters found from Optuna
pipeline.set_params(select__k=trial.params['select_k'],
                    rf__n_estimators=trial.params['n_estimators'],
                    rf__max_depth=trial.params['max_depth'],
                    rf__min_samples_split=trial.params['min_samples_split'],
                    rf__min_samples_leaf=trial.params['min_samples_leaf'],
                    rf__criterion=trial.params['criterion'])

# Fit the pipeline on the full training set
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy: {:.4f}".format(test_accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Display train and test set sizes
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ------------------------------
# 10b. Add optuna
# ------------------------------

from lightgbm import LGBMClassifier

def lgbm_objective(trial):
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    pipeline_lgbm_opt = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
        ('lgbm', LGBMClassifier(**params))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_lgbm_opt, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Run the tuning
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(lgbm_objective, n_trials=30)

# Store best params
lgbm_best_params = lgbm_study.best_params
print("\n✅ Tuned LightGBM Accuracy: {:.4f}".format(lgbm_study.best_value))
print("Best LightGBM Parameters:")
for k, v in lgbm_best_params.items():
    print(f"  {k}: {v}")


# ------------------------------
# 11. Train and Evaluate LightGBM Classifier
# ------------------------------
from lightgbm import LGBMClassifier

# Reuse the same pipeline setup, replacing only the classifier
#pipeline_lgbm = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # use same k
#    ('lgbm', LGBMClassifier(random_state=42))
#])

pipeline_lgbm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=lgbm_best_params['select_k'])),
    ('lgbm', LGBMClassifier(
        n_estimators=lgbm_best_params['n_estimators'],
        learning_rate=lgbm_best_params['learning_rate'],
        num_leaves=lgbm_best_params['num_leaves'],
        max_depth=lgbm_best_params['max_depth'],
        min_child_samples=lgbm_best_params['min_child_samples'],
        subsample=lgbm_best_params['subsample'],
        colsample_bytree=lgbm_best_params['colsample_bytree'],
        random_state=42
    ))
])

# Fit LGBM pipeline
pipeline_lgbm.fit(X_train, y_train)

# Predict
y_pred_lgbm = pipeline_lgbm.predict(X_test)

# Evaluate
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
print("\nLightGBM Test Accuracy: {:.4f}".format(lgbm_accuracy))
print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm))

# Confusion Matrix for LightGBM
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)


# ------------------------------
# 12. Plot Confusion Matrices Side-by-Side
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()


# ------------------------------
# 14. SVM
# ------------------------------

from sklearn.svm import SVC

# Build the same pipeline with SVM
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # reuse k
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# Fit pipeline
pipeline_svm.fit(X_train, y_train)

# Predict
y_pred_svm = pipeline_svm.predict(X_test)

# Evaluate
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Test Accuracy: {:.4f}".format(svm_accuracy))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)


#from catboost import CatBoostClassifier

#pipeline_cat = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),
#    ('catboost', CatBoostClassifier(
#        verbose=0,
#        iterations=300,
#        learning_rate=0.1,
#        depth=6,
#        random_state=42
#    ))
#])

#pipeline_cat.fit(X_train, y_train)
#y_pred_cat = pipeline_cat.predict(X_test)
#cat_accuracy = accuracy_score(y_test, y_pred_cat)
#print("\nCatBoost Test Accuracy: {:.4f}".format(cat_accuracy))
#print("\nCatBoost Classification Report:\n", classification_report(y_test, y_pred_cat))

# Confusion matrix
#conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[2])
axes[2].set_title(f"CatBoost (Acc: {svm_accuracy:.2f})")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# ------------------------------
# 15. Accuracy Comparison Bar Chart
# ------------------------------
#model_names = ['Random Forest', 'LightGBM', 'SVM']
#accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

model_names = ['Random Forest', 'LightGBM', 'SVM']
accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='Set2')
plt.ylim(0.70, 0.85)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xlabel("Model")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.3f}", ha='center', va='bottom', fontsize=10)
plt.show()# ------------------------------
# 0. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for model building and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Imbalanced-learn for oversampling and pipeline (ensures oversampling is applied within CV only)
#%pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
#!pip install optuna
import optuna


# (Optional) If running in Google Colab, mount your Google Drive:
#%pip install google-colab
#try:
#    try:
#        from google.colab import drive
#    except ImportError:
#        drive = None
#    drive.mount('/content/drive')
#except ImportError:
#    print("Not running in Google Colab. Skipping Google Drive mounting.")

# ------------------------------
# 1. Load and Preprocess Data
# ------------------------------
# Adjust file_path to your data location in your Google Drive
file_path = r"C:\Users\PCAdmin\Documents\Projects\MELRE_PricePrediction\Melbourne_RealEstate.csv"
#file_path = 'drive/MyDrive/Colab Notebooks/Melbourne_RealEstate.csv'
df = pd.read_csv(file_path)

# Drop columns that aren’t needed
if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

# ------------------------------
# 2. Handle Missing Values & Outliers
# ------------------------------

# Define column types
numeric_cols = ['Rooms', 'Price', 'Bedroom2', 'Bathroom', 'Car', 'Distance',
                'Propertycount', 'Landsize', 'BuildingArea']
categorical_cols = ['Method', 'Type', 'SellerG', 'Regionname', 'CouncilArea']
date_cols = ['Date']

# Fill missing numeric values with the median
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with "Unknown"
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

# Process date columns: convert to datetime, fill missing values using forward fill
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        df[col].fillna(method='ffill', inplace=True)

# Cap outliers in selected columns at the 99th percentile
outlier_cols = ['Price', 'Landsize', 'BuildingArea']
for col in outlier_cols:
    if col in df.columns:
        upper_limit = df[col].quantile(0.99)
        df.loc[df[col] > upper_limit, col] = upper_limit

# ------------------------------
# 3. Enhanced Feature Engineering
# ------------------------------
# Create date-derived features (Year, Month, Day)
if 'Date' in df.columns:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

# Example: Create an interaction feature using BuildingArea and Landsize (avoid division by zero)
df['AreaRatio'] = np.where(df['Landsize'] > 0, df['BuildingArea'] / df['Landsize'], 0)

# (Optional) If a 'YearBuilt' column exists, compute property Age at sale using the 'Year' feature.
if 'YearBuilt' in df.columns:
    df['Age'] = df['Year'] - df['YearBuilt']
    df['Age'] = df['Age'].apply(lambda x: x if x > 0 else 0)

# Create additional interaction features: e.g., Rooms per Bedroom
if 'Bedroom2' in df.columns:
    df['RoomsPerBedroom'] = np.where(df['Bedroom2'] > 0, df['Rooms'] / df['Bedroom2'], df['Rooms'])

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------
# Use get_dummies for categorical variables (drop_first to avoid collinearity)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# For high-cardinality features such as Suburb use Label Encoding if available
if 'Suburb' in df.columns:
    le = LabelEncoder()
    df['Suburb'] = le.fit_transform(df['Suburb'])

# ------------------------------
# 5. Create Classification Target (Price Category)
# ------------------------------
# Use quantile thresholds to create a categorical variable:
q1 = df['Price'].quantile(0.33)
q2 = df['Price'].quantile(0.67)

def classify_price(price):
    if price <= q1:
        return 0  # Low price
    elif price <= q2:
        return 1  # Medium price
    else:
        return 2  # High price

df['PriceCategory'] = df['Price'].apply(classify_price)
# Remove original Price from features
df.drop('Price', axis=1, inplace=True)

# ------------------------------
# 6. Define Features and Target
# ------------------------------
target = 'PriceCategory'
X = df.drop(columns=[target])
y = df[target]

# Fill any remaining missing values (if any) in features with 0
X.fillna(0, inplace=True)

# ------------------------------
# 7. Split Data into Train/Test sets
# ------------------------------
# Stratify split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# ------------------------------
# 8. Build an Imbalanced-Learn Pipeline
# ------------------------------
# The pipeline includes:
#  - StandardScaler for numerical feature scaling
#  - SMOTE for oversampling the minority classes (applied after scaling)
#  - SelectKBest for feature selection (we let k be optimized)
#  - RandomForestClassifier for prediction

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
])

# ------------------------------
# 9. Hyperparameter Tuning with Optuna
# ------------------------------
def objective(trial):
    # Hyperparameters for SelectKBest: number of features to select (between 5 and total features)
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    # Hyperparameters for Random Forest
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Set parameters for the pipeline steps
    pipeline.set_params(select__k=k,
                          rf__n_estimators=n_estimators,
                          rf__max_depth=max_depth,
                          rf__min_samples_split=min_samples_split,
                          rf__min_samples_leaf=min_samples_leaf,
                          rf__criterion=criterion)

    # Use Stratified K-Fold Cross-Validation for robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

    # Return the average accuracy across folds
    return scores.mean()

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Increase number of trials if computationally feasible

print("Best trial:")
trial = study.best_trial
print("  Accuracy: {:.4f}".format(trial.value))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# ------------------------------
# 10. Train Final Model with Best Hyperparameters & Evaluate
# ------------------------------
# Set the pipeline with the best hyperparameters found from Optuna
pipeline.set_params(select__k=trial.params['select_k'],
                    rf__n_estimators=trial.params['n_estimators'],
                    rf__max_depth=trial.params['max_depth'],
                    rf__min_samples_split=trial.params['min_samples_split'],
                    rf__min_samples_leaf=trial.params['min_samples_leaf'],
                    rf__criterion=trial.params['criterion'])

# Fit the pipeline on the full training set
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy: {:.4f}".format(test_accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Display train and test set sizes
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ------------------------------
# 10b. Add optuna
# ------------------------------

from lightgbm import LGBMClassifier

def lgbm_objective(trial):
    k = trial.suggest_int('select_k', 5, X_train.shape[1])

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    pipeline_lgbm_opt = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
        ('lgbm', LGBMClassifier(**params))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_lgbm_opt, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Run the tuning
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(lgbm_objective, n_trials=30)

# Store best params
lgbm_best_params = lgbm_study.best_params
print("\n✅ Tuned LightGBM Accuracy: {:.4f}".format(lgbm_study.best_value))
print("Best LightGBM Parameters:")
for k, v in lgbm_best_params.items():
    print(f"  {k}: {v}")


# ------------------------------
# 11. Train and Evaluate LightGBM Classifier
# ------------------------------
from lightgbm import LGBMClassifier

# Reuse the same pipeline setup, replacing only the classifier
#pipeline_lgbm = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # use same k
#    ('lgbm', LGBMClassifier(random_state=42))
#])

pipeline_lgbm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=lgbm_best_params['select_k'])),
    ('lgbm', LGBMClassifier(
        n_estimators=lgbm_best_params['n_estimators'],
        learning_rate=lgbm_best_params['learning_rate'],
        num_leaves=lgbm_best_params['num_leaves'],
        max_depth=lgbm_best_params['max_depth'],
        min_child_samples=lgbm_best_params['min_child_samples'],
        subsample=lgbm_best_params['subsample'],
        colsample_bytree=lgbm_best_params['colsample_bytree'],
        random_state=42
    ))
])

# Fit LGBM pipeline
pipeline_lgbm.fit(X_train, y_train)

# Predict
y_pred_lgbm = pipeline_lgbm.predict(X_test)

# Evaluate
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
print("\nLightGBM Test Accuracy: {:.4f}".format(lgbm_accuracy))
print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm))

# Confusion Matrix for LightGBM
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)


# ------------------------------
# 12. Plot Confusion Matrices Side-by-Side
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()


# ------------------------------
# 14. SVM
# ------------------------------

from sklearn.svm import SVC

# Build the same pipeline with SVM
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),  # reuse k
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# Fit pipeline
pipeline_svm.fit(X_train, y_train)

# Predict
y_pred_svm = pipeline_svm.predict(X_test)

# Evaluate
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Test Accuracy: {:.4f}".format(svm_accuracy))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)


#from catboost import CatBoostClassifier

#pipeline_cat = Pipeline([
#    ('scaler', StandardScaler()),
#    ('smote', SMOTE(random_state=42)),
#    ('select', SelectKBest(score_func=mutual_info_classif, k=trial.params['select_k'])),
#    ('catboost', CatBoostClassifier(
#        verbose=0,
#        iterations=300,
#        learning_rate=0.1,
#        depth=6,
#        random_state=42
#    ))
#])

#pipeline_cat.fit(X_train, y_train)
#y_pred_cat = pipeline_cat.predict(X_test)
#cat_accuracy = accuracy_score(y_test, y_pred_cat)
#print("\nCatBoost Test Accuracy: {:.4f}".format(cat_accuracy))
#print("\nCatBoost Classification Report:\n", classification_report(y_test, y_pred_cat))

# Confusion matrix
#conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[0])
axes[0].set_title(f"Random Forest (Acc: {test_accuracy:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(conf_matrix_lgbm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[1])
axes[1].set_title(f"LightGBM (Acc: {lgbm_accuracy:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"], ax=axes[2])
axes[2].set_title(f"CatBoost (Acc: {svm_accuracy:.2f})")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# ------------------------------
# 15. Accuracy Comparison Bar Chart
# ------------------------------
#model_names = ['Random Forest', 'LightGBM', 'SVM']
#accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

model_names = ['Random Forest', 'LightGBM', 'SVM']
accuracies = [test_accuracy, lgbm_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='Set2')
plt.ylim(0.70, 0.85)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xlabel("Model")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.3f}", ha='center', va='bottom', fontsize=10)
plt.show()
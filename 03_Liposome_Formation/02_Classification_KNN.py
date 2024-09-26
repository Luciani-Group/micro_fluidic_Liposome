import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os

# Load the data from the Excel file
file_path = r'C:\Users\Remo E\OneDrive - Universitaet Bern\General - PhD\001_Experiments\RE_050\20240517_Liposome_data.xlsx'
data = pd.read_excel(file_path)

# Specify the target column and the feature columns
target_column = 'Population'  # actual target column name
feature_columns = ['CHOL %', 'DSPE-PEG %', 'CHIP', 'FRR', 'Chain length 1', 'Unsatturation chain 1', 'Chain length 2', 'Unsatturation chain 2']
stratify_column = 'Lipid'  # column name for stratified sampling

# Split features and target
X = data[feature_columns]
y = data[target_column]

# Convert target variable to binary: 2 -> 0, 1 -> 1
y_binary = np.where(y == 2, 0, y)

# Define colors for the ROC curves
roc_colors = ['#9b59b6', '#8e44ad', '#7d3c98', '#6c3483', '#5b2c6f']
mean_color = '#1f77b4'

# Initialize the cross-validation splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define a function to evaluate the model
def evaluate_model(y_test, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Initialize a dictionary to store results for each set of features
results = {}

# Iterate over the number of features to include
for i in range(1, len(feature_columns) + 1):
    feature_subset = feature_columns[:i]
    print(f"Evaluating model with features: {feature_subset}")
    
    # Split features and target
    X_subset = data[feature_subset]
    y = data[target_column]

    # Define a column transformer to apply different transformations to different columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first'), ['CHIP']) if 'CHIP' in feature_subset else ('pass', 'passthrough', feature_subset)
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )

    # Apply the transformations
    X_transformed = column_transformer.fit_transform(X_subset)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)

    # Initialize lists for predictions
    tprs = []
    aucs = []
    y_tests = []
    y_preds = []
    y_probs = []

    # Evaluation results storage
    evaluation_results = []

    # Iterate over each fold
    for train_idx, test_idx in cv.split(X_scaled, y_binary):
        # Split the data into training and test sets for this fold
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # Train a K-Nearest Neighbors classifier
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)

        # Get the probability scores for the test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Store predictions
        y_tests.extend(y_test)
        y_preds.extend(y_pred)
        y_probs.extend(y_prob)

        # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) for various threshold values
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Calculate the Area Under the ROC Curve (AUC) for this fold
        roc_auc = auc(fpr, tpr)

        # Interpolate the ROC curve for calculating the mean ROC curve
        interp_tpr = np.interp(np.linspace(0, 1, 100), fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

        # Evaluate the model and store the results
        evaluation_results.append(evaluate_model(y_test, y_pred, y_prob))

    # Calculate the mean and std of AUC
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Store results
    results[f"Features {i}"] = {
        'features': feature_subset,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'evaluation_results': evaluation_results
    }

    # Print results for the current set of features
    print(f"Features: {feature_subset}, Mean AUC: {mean_auc:.4f}, Std AUC: {std_auc:.4f}")

# Ensure the output directory exists
output_dir = r'D:\Remo\Porgrams\Codes_VSC\Lipo_AI\Model_Evaluations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save evaluation results to an Excel file in the specified directory
output_file_path = os.path.join(output_dir, 'KNN_incremental_features_evaluation.xlsx')
with pd.ExcelWriter(output_file_path) as writer:
    for key, result in results.items():
        df = pd.DataFrame(result['evaluation_results'])
        df.to_excel(writer, sheet_name=key, index_label='Fold')
        # Add a note with feature names used
        df_features = pd.DataFrame(result['features'], columns=['Features Used'])
        df_features.to_excel(writer, sheet_name=key, startrow=df.shape[0] + 2, index=False)

# Plot the mean ROC curve for the final set of features
fig, ax = plt.subplots(figsize=(7, 5))
mean_tpr = np.mean(tprs, axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)
ax.plot(np.linspace(0, 1, 100), mean_tpr, color=mean_color, label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

# Plot the random guess line
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess')

# Configure plot
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Receiver Operating Characteristic - Cross-Validation', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.show()

# Plot the 5x cross-validation ROC curves for the model with all 8 features
final_tprs = []
final_aucs = []

# Split features and target
X_final = data[feature_columns]
y = data[target_column]

# Define a column transformer to apply different transformations to different columns
column_transformer_final = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['CHIP']) if 'CHIP' in feature_columns else ('pass', 'passthrough', feature_columns)
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

# Apply the transformations
X_transformed_final = column_transformer_final.fit_transform(X_final)

# Standardize the features
X_scaled_final = scaler.fit_transform(X_transformed_final)

fig, ax = plt.subplots(figsize=(7, 5))

for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled_final, y_binary)):
    X_train, X_test = X_scaled_final[train_idx], X_scaled_final[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, lw=1, alpha=0.3, color=roc_colors[i], label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')
    
    interp_tpr = np.interp(np.linspace(0, 1, 100), fpr, tpr)
    interp_tpr[0] = 0.0
    final_tprs.append(interp_tpr)
    final_aucs.append(roc_auc)

# Plot the mean ROC curve
mean_tpr_final = np.mean(final_tprs, axis=0)
mean_auc_final = np.mean(final_aucs)
std_auc_final = np.std(final_aucs)
ax.plot(np.linspace(0, 1, 100), mean_tpr_final, color=mean_color, label=f'Mean ROC (AUC = {mean_auc_final:.2f} ± {std_auc_final:.2f})', lw=2)

# Plot the random guess line
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess')

# Configure plot
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Receiver Operating Characteristic - Cross-Validation with All Features', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.show()

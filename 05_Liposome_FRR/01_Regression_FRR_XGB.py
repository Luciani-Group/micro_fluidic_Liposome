import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

import random 
random.seed(42)
np.random.seed(42)

# Load the data from the Excel file
file_path = r'C:\Users\Remo E\OneDrive - Universitaet Bern\General - PhD\001_Experiments\RE_050\20240517_Liposome_data.xlsx'
data = pd.read_excel(file_path)

# Exclude rows where 'Population' is 2
data = data[data['Population'] != 2].reset_index(drop=True)

# Specify the target column and the feature columns
target_column = 'FRR'  # Replace with your actual target column name
feature_columns = ['CHOL %', 'DSPE-PEG %', 'CHIP', 'Size', 'Chain length 1', 'Unsatturation chain 1', 'Chain length 2', 'Unsatturation chain 2']

# Split features and target
X = data[feature_columns]
y = data[target_column]

# Define a function to evaluate the model
def evaluate_model(y_test, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

# Initialize a dictionary to store results for each set of features
results = {}

# Iterate over the number of features to include
for i in range(1, len(feature_columns) + 1):
    feature_subset = feature_columns[:i]
    print(f"Evaluating model with features: {feature_subset}")

    # Split features and target
    X_subset = data[feature_subset]

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
    y_tests = []
    y_preds = []

    # Evaluation results storage
    evaluation_results = []

    # Initialize the cross-validation splitter
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate over each fold
    for train_idx, test_idx in cv.split(X_scaled):
        # Split the data into training and test sets for this fold
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train an XGBoost regressor
        model = xgb.XGBRegressor(objective='reg:squarederror',    alpha=0.1,  # L1 regularization
    )
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Store predictions
        y_tests.extend(y_test)
        y_preds.extend(y_pred)

        # Evaluate the model and store the results
        evaluation_results.append(evaluate_model(y_test, y_pred))

    # Calculate the mean and std of the evaluation metrics
    results_df = pd.DataFrame(evaluation_results)
    mean_results = results_df.mean().to_dict()
    std_results = results_df.std().to_dict()

    # Store results
    results[f"Features {i}"] = {
        'features': feature_subset,
        'mean_results': mean_results,
        'std_results': std_results,
        'evaluation_results': evaluation_results
    }

    # Print results for the current set of features
    print(f"Features: {feature_subset}, Mean Results: {mean_results}, Std Results: {std_results}")

# Ensure the output directory exists
output_dir = r'D:\Remo\Porgrams\Codes_VSC\Lipo_AI\Model_Evaluations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save evaluation results to an Excel file in the specified directory
output_file_path = os.path.join(output_dir, 'XGB_incremental_features_evaluation_regression_FRR.xlsx')
if os.path.exists(output_file_path):
    os.remove(output_file_path)

with pd.ExcelWriter(output_file_path, mode='w') as writer:
    for key, result in results.items():
        df = pd.DataFrame(result['evaluation_results'])
        df.to_excel(writer, sheet_name=key, index_label='Fold')
        # Add a note with feature names used
        df_features = pd.DataFrame(result['features'], columns=['Features Used'])
        df_features.to_excel(writer, sheet_name=key, startrow=df.shape[0] + 2, index=False)

# Initialize the cross-validation splitter for SHAP and final model evaluation
cv_shap = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store SHAP values and test sets for the final model
shap_values_list = []
X_test_list = []

# Initialize lists for final model evaluation metrics
final_results = []

# Split features and target for the final model
X_final = data[feature_columns]
y_final = data[target_column]

# Apply the transformations for the final model
X_transformed_final = column_transformer.fit_transform(X_final)

# Standardize the features for the final model
X_scaled_final = scaler.fit_transform(X_transformed_final)

# Get the transformed feature names for the final model
onehot_feature_names = column_transformer.transformers_[0][1].get_feature_names_out(['CHIP']) if 'CHIP' in feature_columns else []
other_feature_names = [f for f in feature_columns if f != 'CHIP']
all_feature_names = np.concatenate([onehot_feature_names, other_feature_names])

# Initialize lists to store true and predicted values for the final model
all_y_test = []
all_y_pred = []

for train_idx, test_idx in cv_shap.split(X_scaled_final):
    X_train, X_test = X_scaled_final[train_idx], X_scaled_final[test_idx]
    y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]

    # Train the final XGBoost regressor
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Store predictions
    final_results.append(evaluate_model(y_test, y_pred))
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

    # Explain the model predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap_values_list.append(shap_values)
    X_test_list.append(X_test)

# Print evaluation results for the final model
final_results_df = pd.DataFrame(final_results)
print(final_results_df)

# Save final evaluation results to an Excel file
final_output_file_path = os.path.join(output_dir, 'XGB_final_evaluation_regression_FRR.xlsx')
if os.path.exists(final_output_file_path):
    os.remove(final_output_file_path)

with pd.ExcelWriter(final_output_file_path, engine='openpyxl', mode='w') as writer:
    final_results_df.to_excel(writer, sheet_name='Final Evaluation', index_label='Fold')

# Save the trained model
model_output_dir = r'D:\Remo\Porgrams\Codes_VSC\Lipo_AI\Models'
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
model_output_path = os.path.join(model_output_dir, 'xgb_FRR_model.json')
model.save_model(model_output_path)

# Concatenate SHAP values from all folds
shap_values_concat = np.concatenate([sv.values for sv in shap_values_list], axis=0)

# Concatenate corresponding X_test sets from all folds
X_test_concat = np.concatenate(X_test_list, axis=0)

# Plot SHAP summary plot with feature column names
shap.summary_plot(shap_values_concat, features=X_test_concat, feature_names=all_feature_names)

# Plot predicted FRR against actual FRR
plt.figure(figsize=(10, 6))
plt.scatter(all_y_test, all_y_pred, alpha=0.3)
plt.plot([min(all_y_test), max(all_y_test)], [min(all_y_test), max(all_y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual FRR', fontsize=16, fontweight='bold')
plt.ylabel('Predicted FRR', fontsize=16, fontweight='bold')
plt.title('Actual FRR vs Predicted FRR', fontsize=16, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the raw data for the plot to the Excel file
raw_data_df = pd.DataFrame({'Actual Size': all_y_test, 'Predicted Size': all_y_pred})
with pd.ExcelWriter(final_output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    raw_data_df.to_excel(writer, sheet_name='Raw Data', index=False)

# Apply the same column transformer to the data for the Spearman's correlation heatmap
data_transformed = column_transformer.fit_transform(data[feature_columns])
transformed_feature_names = np.concatenate([onehot_feature_names, other_feature_names])

# Convert the transformed data to a DataFrame
data_transformed_df = pd.DataFrame(data_transformed, columns=transformed_feature_names)

# Calculate and plot Spearman's correlation heatmap for all features
correlation_matrix = data_transformed_df.corr(method='spearman')
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Spearman's Correlation Heatmap", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Save the correlation matrix to the Excel file
with pd.ExcelWriter(final_output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')

# clustering with FRR using PCA and t-SNE
pca = PCA(n_components=5)
shap_pca50 = pca.fit_transform(shap_values_concat)

tsne = TSNE(n_components=2, perplexity=50)
shap_embedded = tsne.fit_transform(shap_pca50)

cdict1 = {
    "red": (
        (0.0, 0.11764705882352941, 0.11764705882352941),
        (1.0, 0.9607843137254902, 0.9607843137254902),
    ),
    "green": (
        (0.0, 0.5333333333333333, 0.5333333333333333),
        (1.0, 0.15294117647058825, 0.15294117647058825),
    ),
    "blue": (
        (0.0, 0.8980392156862745, 0.8980392156862745),
        (1.0, 0.3411764705882353, 0.3411764705882353),
    ),
    "alpha": ((0.0, 1, 1), (0.5, 1, 1), (1.0, 1, 1)),
}  # #1E88E5 -> #ff0052
red_blue_solid = LinearSegmentedColormap("RedBlue", cdict1)

plt.figure(figsize=(5, 5))
plt.scatter(
    shap_embedded[:, 0],
    shap_embedded[:, 1],
    c=all_y_pred[:1000],  # Color by predicted FRR values
    linewidth=0,
    alpha=1.0,
    cmap=red_blue_solid,
)
cb = plt.colorbar(label="Predicted FRR value", aspect=40, orientation="horizontal")
cb.set_alpha(1)
cb.outline.set_linewidth(0)
cb.ax.tick_params("x", length=0)
cb.ax.xaxis.set_label_position("top")
plt.gca().axis("off")
plt.title("Supervised Clustering with Predicted FRR", fontsize=16, fontweight='bold')
plt.show()

# Accessing the KL divergence after fitting t-SNE
kl_divergence = tsne.kl_divergence_
print(f"KL Divergence: {kl_divergence}")

# Calculate residuals
residuals = np.array(all_y_test) - np.array(all_y_pred)

# Plot residuals vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(all_y_pred, residuals, alpha=0.5)
plt.hlines(0, min(all_y_pred), max(all_y_pred), colors='red', linestyles='dashed')
plt.xlabel('Predicted FRR ', fontsize=16, fontweight='bold')
plt.ylabel('Residuals', fontsize=16, fontweight='bold')
plt.title('Residuals vs. Predicted Size', fontsize=16, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame to store the actual values, predicted values, and residuals
residuals_df = pd.DataFrame({
    'Actual Size': all_y_test,
    'Predicted Size': all_y_pred,
    'Residuals': residuals
})

# Define the path for the output Excel file
residuals_output_file_path = os.path.join(output_dir, 'XGB_final_evaluation_regression_FRR.xlsx')

# Save the residuals DataFrame to an Excel file
with pd.ExcelWriter(residuals_output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    residuals_df.to_excel(writer, sheet_name='Residuals', index=False)



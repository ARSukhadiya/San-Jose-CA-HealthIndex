# %%
import pandas as pd
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data/NeighborhoodData_finalData_2.csv')
neighborhood_data = pd.read_csv(data_path)

# %%
# neighborhood_data = pd.read_csv('app/data/NeighborhoodData_finalData_2.csv')

# %%
neighborhood_data.columns

# %%
neighborhood_data.head(5)

# %%
neighborhood_data.Health_outcome.unique()

# %%
neighborhood_data.drop(['Unnamed: 0','NGBRHD2'],axis=1,inplace=True)

# %%
neighborhood_data.head(5)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Basic statistics for initial data understanding
eda_summary = neighborhood_data.describe()
eda_summary

# %%
# Check for missing values
missing_values = neighborhood_data.isnull().sum()
missing_values

# %%
# Scatter Plots in 2x3 layout
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# fig.suptitle("Scatter Plots of Key Feature Relationships", fontsize=16)

# scatter_features = [('medhhinc', 'obese_adult'), ('medhhinc', 'unemp'), ('unemp', 'birth_lw'),
#                     ('medhhinc', 'ed_hsgrad'), ('chldfpl185', 'birth_lw'), ('medhhinc', 'overobese_teen')]
# for i, (x, y) in enumerate(scatter_features):
#     row, col = divmod(i, 3)
#     sns.scatterplot(data=neighborhood_data, x=x, y=y, hue='Health_outcome', ax=axes[row, col])
#     axes[row, col].set_title(f"{x} vs {y}")
#     axes[row, col].set_xlabel(x)
#     axes[row, col].set_ylabel(y)
#     axes[row, col].legend(title='Health Outcome', loc='upper right')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# %%
# Histograms in 2x3 layout
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# fig.suptitle("Histograms of Key Features", fontsize=16)

# hist_features = ['medhhinc', 'obese_adult', 'unemp', 'birth_lw', 'ed_hsgrad', 'chldfpl185']
# for i, feature in enumerate(hist_features):
#     row, col = divmod(i, 3)
#     sns.histplot(neighborhood_data[feature], kde=True, color='teal', ax=axes[row, col])
#     axes[row, col].set_title(f"Distribution of {feature}")
#     axes[row, col].set_xlabel(feature)
#     axes[row, col].set_ylabel("Frequency")

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# %%
# Box Plots in 2x3 layout with adjusted legend handling
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# fig.suptitle("Box Plots of Key Features by Health Outcome", fontsize=16)

# box_features = ['medhhinc', 'obese_adult', 'unemp', 'birth_lw', 'ed_hsgrad', 'chldfpl185']
# for i, feature in enumerate(box_features):
#     row, col = divmod(i, 3)
#     sns.boxplot(data=neighborhood_data, x='Health_outcome', y=feature, hue='Health_outcome', palette='viridis', ax=axes[row, col], dodge=False)
#     axes[row, col].set_title(f"{feature} by Health Outcome")
#     axes[row, col].set_xlabel("Health Outcome")
#     axes[row, col].set_ylabel(feature)
#     if axes[row, col].get_legend():
#         axes[row, col].get_legend().remove()  # Remove redundant legends for each subplot

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# %%
# Adjusted code with checks for legend existence before attempting to remove it

# Define figure and axes for box and violin plots with adjustments
# fig, axes = plt.subplots(3, 2, figsize=(10, 12))
# fig.suptitle("Feature Relationships with Health Outcome (Target Variable)", fontsize=20)

# # Box plot for Median Household Income (medhhinc) by Health Outcome
# sns.boxplot(data=neighborhood_data, x='Health_outcome', y='medhhinc', hue='Health_outcome', palette='coolwarm', ax=axes[0, 0], dodge=False)
# axes[0, 0].set_title("Median Household Income by Health Outcome")
# axes[0, 0].set_xlabel("Health Outcome")
# axes[0, 0].set_ylabel("Median Household Income")
# if axes[0, 0].get_legend():
#     axes[0, 0].legend_.remove()  # Remove redundant legend if present

# # Box plot for Obesity Levels (obese_adult) by Health Outcome
# sns.boxplot(data=neighborhood_data, x='Health_outcome', y='obese_adult', hue='Health_outcome', palette='viridis', ax=axes[0, 1], dodge=False)
# axes[0, 1].set_title("Adult Obesity Levels by Health Outcome")
# axes[0, 1].set_xlabel("Health Outcome")
# axes[0, 1].set_ylabel("Obesity Level (%)")
# if axes[0, 1].get_legend():
#     axes[0, 1].legend_.remove()  # Remove redundant legend if present

# # Violin plot for Unemployment Rates by Health Outcome
# sns.violinplot(data=neighborhood_data, x='Health_outcome', y='unemp', hue='Health_outcome', palette='Spectral', ax=axes[1, 0], dodge=False)
# axes[1, 0].set_title("Unemployment Rates by Health Outcome")
# axes[1, 0].set_xlabel("Health Outcome")
# axes[1, 0].set_ylabel("Unemployment Rate (%)")
# if axes[1, 0].get_legend():
#     axes[1, 0].legend_.remove()  # Remove redundant legend if present

# # Violin plot for Low Birth Weight Rates (birth_lw) by Health Outcome
# sns.violinplot(data=neighborhood_data, x='Health_outcome', y='birth_lw', hue='Health_outcome', palette='rocket', ax=axes[1, 1], dodge=False)
# axes[1, 1].set_title("Low Birth Weight Rates by Health Outcome")
# axes[1, 1].set_xlabel("Health Outcome")
# axes[1, 1].set_ylabel("Low Birth Weight Rate (%)")
# if axes[1, 1].get_legend():
#     axes[1, 1].legend_.remove()  # Remove redundant legend if present

# # Bar plot for Education Level (High School Graduates) by Health Outcome
# sns.barplot(data=neighborhood_data, x='Health_outcome', y='ed_hsgrad', hue='Health_outcome', palette='magma', ax=axes[2, 0], dodge=False)
# axes[2, 0].set_title("High School Graduation Rates by Health Outcome")
# axes[2, 0].set_xlabel("Health Outcome")
# axes[2, 0].set_ylabel("High School Graduation Rate (%)")
# if axes[2, 0].get_legend():
#     axes[2, 0].legend_.remove()  # Remove redundant legend if present

# # Bar plot for Percentage of Families Below Poverty Line (famfpl185) by Health Outcome
# sns.barplot(data=neighborhood_data, x='Health_outcome', y='famfpl185', hue='Health_outcome', palette='icefire', ax=axes[2, 1], dodge=False)
# axes[2, 1].set_title("Percentage of Families Below Poverty Line by Health Outcome")
# axes[2, 1].set_xlabel("Health Outcome")
# axes[2, 1].set_ylabel("Families Below Poverty Line (%)")
# if axes[2, 1].get_legend():
#     axes[2, 1].legend_.remove()  # Remove redundant legend if present

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


# %%
neighborhood_data.OBJECTID_x.nunique(),neighborhood_data.OBJECTID_y.nunique()

# %%
neighborhood_data.drop(['OBJECTID_x','OBJECTID_y'],axis=1,inplace=True)

# %%
neighborhood_data.columns

# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %%
label_encoder = LabelEncoder()
neighborhood_data['Health_outcome_encoded'] = label_encoder.fit_transform(neighborhood_data['Health_outcome'])
X = neighborhood_data.drop(columns=['Health_outcome','Health_outcome_encoded'])
y = neighborhood_data['Health_outcome_encoded']

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)
# Split the data with stratification on the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% [markdown]
# ## RandomForest Classifier

# %%
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])
pipeline

# %%
# Define hyperparameter grid for GridSearchCV
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10]
}

# Set up GridSearchCV with the RandomForestClassifier
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search

# %%
grid_search.fit(X_train, y_train)

# %%
# Output the best parameters and best score
best_params = grid_search.best_params_
best_params

# %%
best_score = grid_search.best_score_
best_score

# %%
# Make predictions on the test set and generate a classification report
y_pred = grid_search.predict(X_test)

# %%
grid_search.best_estimator_.score(X_test,y_test)

# %% [markdown]
# ## XgBoost Classifier

# %%
# Define a pipeline with StandardScaler and XGBClassifier
from xgboost import XGBClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(random_state=42,eval_metric='mlogloss'))
])

# Define hyperparameter grid for XGBClassifier
param_grid = {
    'xgb__n_estimators': [50, 100, 150],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.8, 1.0]
}

# %%
# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Output best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_params

# %%
accuracy= grid_search.best_estimator_.score(X_test,y_test)
accuracy


import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

# Assuming grid_search.best_estimator_ is the best trained model
# Generate predictions and predicted probabilities for X_test
y_pred = grid_search.best_estimator_.predict(X_test)
y_proba = grid_search.best_estimator_.predict_proba(X_test)

# Binarize the output for ROC AUC (assuming y_test has classes 0, 1, and 2)
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])

# Calculate ROC AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
roc_auc_plot = go.Figure()

for i in range(y_test_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    class_name = [key for key, value in label_mapping.items() if value == i][0]
    roc_auc_plot.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'Class: {class_name} (AUC = {roc_auc[i] * 100:.0f}%)'))

# Add a reference line for a random classifier
roc_auc_plot.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name="Random Classifier"))
roc_auc_plot.update_layout(
    title="ROC AUC Curves by Class",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend_title="Classes"
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=grid_search.best_estimator_.classes_)

# Define the labels for the axes
labels = ['Healthy', 'At Risk', 'Chronic']

# Create the heatmap with annotations
conf_matrix_plot = go.Figure(data=go.Heatmap(
    z=cm, 
    x=labels, 
    y=labels, 
    colorscale='Blues', 
    hoverongaps=False,
    showscale=True
))

# Add annotations for each cell in the confusion matrix
for i in range(len(cm)):
    for j in range(len(cm[i])):
        conf_matrix_plot.add_annotation(
            x=labels[j],
            y=labels[i],
            text=str(cm[i][j]),
            showarrow=False,
            font=dict(color="black", size=12)  # Customize font color and size if needed
        )

# Update layout for axis titles and overall plot title
conf_matrix_plot.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)


# Calculate overall metrics
accuracy = grid_search.best_estimator_.score(X_test, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc_overall = roc_auc_score(y_test_binarized, y_proba, average='weighted', multi_class="ovr")
colors = ['skyblue', 'salmon', 'khaki', 'peachpuff', 'plum']


# Create a bar plot for these metrics
metrics_plot = go.Figure(data=[go.Bar(
    x=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    y=[accuracy, precision, recall, f1, roc_auc_overall],
    marker_color=colors,
    text=[f"{accuracy * 100:.0f}%", f"{precision * 100:.0f}%", f"{recall * 100:.0f}%", f"{f1 * 100:.0f}%", f"{roc_auc_overall * 100:.0f}%"],
    textposition='auto'
)])
metrics_plot.update_layout(title="Performance Metrics- Score %")

roc_auc_plot.update_layout(height=600,width=800)
conf_matrix_plot.update_layout(height=600,width=800)
metrics_plot.update_layout(height=600,width=800)


import plotly.io as pio
import os

# Directory to save plots
plot_dir = "static/plots_metrics"
os.makedirs(plot_dir, exist_ok=True)

# Save each plot as a JSON file
pio.write_json(roc_auc_plot, f"{plot_dir}/roc_auc_plot.json")
pio.write_json(conf_matrix_plot, f"{plot_dir}/conf_matrix_plot.json")
pio.write_json(metrics_plot, f"{plot_dir}/metrics_plot.json")


# %%
# Generate predictions for calculating precision and recall
# from sklearn.metrics import precision_score, recall_score

# # Generate predictions on the test set
# y_pred = grid_search.best_estimator_.predict(X_test)

# # Calculate precision and recall for each class and overall (macro or weighted average)
# precision = precision_score(y_test, y_pred, average='weighted')  # Options: 'micro', 'macro', 'weighted'
# recall = recall_score(y_test, y_pred, average='weighted')

# print("Accuracy:", grid_search.best_estimator_.score(X_test, y_test))
# print("Precision:", precision)
# print("Recall:", recall)
# import os
# import plotly.io as pio
# from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, accuracy_score
# import plotly.graph_objs as go
# import plotly.io as pio

# # Define a directory for saving plots
# plot_dir = "static/plots"
# os.makedirs(plot_dir, exist_ok=True)

# y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)  # Ensure this is probabilities, not labels
# roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
# conf_matrix = confusion_matrix(y_test, y_pred)
# precision = precision_score(y_test, y_pred,average='macro')
# recall = recall_score(y_test, y_pred,average='macro')
# accuracy = accuracy_score(y_test, y_pred)

# # Generate and save ROC AUC Plot
# roc_plot = go.Figure()
# roc_plot.add_trace(go.Scatter(x=[0, 1], y=[0, roc_auc], mode='lines+markers', name='ROC Curve'))
# roc_plot.update_layout(title="ROC AUC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
# pio.write_image(roc_plot, f"{plot_dir}/roc_auc.png")

# # Generate and save Confusion Matrix Plot
# conf_matrix_plot = go.Figure(data=go.Heatmap(
#     z=conf_matrix,
#     x=['Predicted Negative', 'Predicted Positive'],
#     y=['Actual Negative', 'Actual Positive'],
#     colorscale='Blues'))
# conf_matrix_plot.update_layout(title="Confusion Matrix")
# pio.write_image(conf_matrix_plot, f"{plot_dir}/conf_matrix.png")

# # Generate and save Precision, Recall, Accuracy Bar Plot
# metric_bar = go.Figure(data=[go.Bar(
#     x=['Precision', 'Recall', 'Accuracy'],
#     y=[precision, recall, accuracy],
#     text=[precision, recall, accuracy],
#     textposition='auto'
# )])
# metric_bar.update_layout(title="Performance Metrics")
# pio.write_image(metric_bar, f"{plot_dir}/metrics_bar.png")

# %%


# %%




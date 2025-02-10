#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import contextily as ctx
import folium
import os


# In[5]:


# Load the dataset (uncomment the following four lines after adding your folder path)
#path = '/path/to/your/folder'  # Replace this with your folder path
#os.chdir(path)  # Change the working directory to the specified path
#file_name = 'CA_wildfires.csv'  # Input file name
#data = pd.read_csv(os.path.join(path, file_name))


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Clean column names to remove erroneous spaces and asterisks
data.columns = data.columns.str.replace(r'\s+|\*', '', regex=True)

# Create dummy variable for severe damage
def categorize_damage(damage):
    if damage == 'Destroyed (>50%)':
        return 1  # Severe damage (damage > 50%)
    return 0  # No severe damage

# Apply the categorization function to the 'Damage' column
data['Damage'] = data['Damage'].apply(categorize_damage)
    
# Impute missing values instead of dropping rows
for col in data.columns:
    if data[col].dtype == 'object':
        mode = data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown"
        data[col] = data[col].fillna(mode)  # Fill categorical with mode or "Unknown"
    else:
        median = data[col].median() if not data[col].isna().all() else 0
        data[col] = data[col].fillna(median)  # Fill numerical with median or 0

# Select relevant features and drop unnecessary ones
selected_columns = [
    'RoofConstruction', 'Eaves', 'VentScreen', 'ExteriorSiding',
    'WindowPane', 'Deck/PorchOnGrade', 'Deck/PorchElevated',
    'PatioCover/CarportAttachedtoStructure', 'FenceAttachedtoStructure',
    'AssessedImprovedValue(parcel)', 'YearBuilt(parcel)', 'Latitude', 'Longitude', 'Damage'
]
filtered_data = data[selected_columns]

# Encode categorical variables using one-hot encoding
filtered_data = pd.get_dummies(filtered_data, drop_first=True)

# Separate features and target variable
X = filtered_data.drop(columns=['Damage'])
y = filtered_data['Damage']

# Check if dataset has enough samples
if len(filtered_data) < 1:
    raise ValueError("The dataset is empty after preprocessing")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize features using StandardScaler for kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report dictionary to a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Print the accuracy and classification report as a table
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(report_df)


# In[7]:


# Now let's try a Naive Bayes classifier on the same data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
nb = GaussianNB()
nb.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Predict on the test set
y_pred = nb.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report dictionary to a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Output results
print("Naïve Bayes Model Accuracy:", accuracy)
print("\nClassification Report:")
print(report_df)


# In[8]:


from sklearn.model_selection import cross_val_score

# Inspect distribution of damage
print(data['Damage'].value_counts())

# Perform cross-validation for KNN
cv_scores = cross_val_score(knn, X, y, cv=5)  # 5-fold cross-validation

print("KNN Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

k_values = [1, 2,3, 4,5, 6,7, 8,9]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(f"Accuracy with k={k}: {accuracy_score(y_test, y_pred)}")

# Cross-validation for Naïve Bayes
nb = GaussianNB()
nb_scores = cross_val_score(nb, X, y, cv=5, scoring='accuracy')  # No need to scale for Naïve Bayes
print("Naïve Bayes Mean Accuracy:", nb_scores.mean())

# The KNN classifier is significantly more accurate than the Naive Bayes.


# In[9]:


# Convert heat map columns to lists of floats
def convert_columns_to_float_lists(data, columns):

  float_lists = {}
  for col in columns:
    try:
      float_lists[col] = data[col].astype(float).tolist()
    except ValueError as e:
      print(f"Error converting '{col}' to floats: {e}")
      float_lists[col] = []  # Or handle the error differently

  return float_lists

columns_to_convert = ['Latitude', 'Longitude','Damage']
float_data = convert_columns_to_float_lists(data, columns_to_convert)

print(float_data)


# In[19]:


import folium
import pandas as pd
from folium.plugins import HeatMap

# Load and clean the data
heatmapdata = pd.read_csv('CA_wildfires.csv')
heatmapdata.columns = heatmapdata.columns.str.replace(r'\s+|\*', '', regex=True)

# Define damage mapping function
def map_damage_classifications(damage_value):
    damage_mapping = {
        "No Damage": 1,
        "Affected (1-9%)": 2,
        "Minor (10-25%)": 3,
        "Major (26-50%)": 4,
        "Destroyed (>50%)": 5
    }
    return damage_mapping.get(damage_value, None)

# Apply damage score mapping
if 'Damage' in heatmapdata.columns:
    heatmapdata['DamageScore'] = heatmapdata['Damage'].apply(map_damage_classifications)
else:
    raise ValueError("'Damage' column not found in the dataset.")

# Create scatter plot with heat map
def create_damage_map(data, 
                      location=[34, -118], 
                      zoom_start=10, 
                      output_file="wildfire_damage_map.html"):
    # Filter out rows with None damage scores
    filtered_data = data.dropna(subset=['DamageScore', 'Latitude', 'Longitude'])
    
    # Create base map
    damage_map = folium.Map(location=location, zoom_start=zoom_start)
    
    # Prepare heat map data (using only scores > 1)
    heat_data = filtered_data[filtered_data['DamageScore'] > 1][['Latitude', 'Longitude', 'DamageScore']].values.tolist()
    
    # Add heat map layer
    HeatMap(heat_data, 
            name='Damage Intensity', 
            min_opacity=0.5,
            max_opacity=0.8,
            blur=10).add_to(damage_map)
    
    # Add circle markers for individual properties
    for _, row in filtered_data[filtered_data['DamageScore'] > 1].iterrows():
        color_map = {
            2: 'green',   # Affected
            3: 'yellow',  # Minor
            4: 'orange',  # Major
            5: 'red'      # Destroyed
        }
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['DamageScore'],
            popup=f"Damage Score: {row['DamageScore']}",
            color=color_map.get(row['DamageScore'], 'gray'),
            fill=True,
            fillOpacity=0.7
        ).add_to(damage_map)
    
    # Save the map
    damage_map.save(output_file)
    print(f"Map saved to {output_file}")

# Create the map
create_damage_map(heatmapdata)


# In[43]:


# Assorted graphs

# Countplot for damage categories
plt.figure(figsize=(8, 5))
custom_palette = {
    1: 'blue',   # No Damage
    2: 'green',  # Affected
    3: 'yellow', # Minor
    4: 'orange', # Major
    5: 'red'     # Destroyed
}
# Convert custom palette to a list matching the x-axis order
palette = [custom_palette[i] for i in range(1, 6)]

sns.countplot(x='DamageScore', data=heatmapdata, palette=palette)
plt.title('Distribution of Damage Categories', fontsize=16)
plt.xlabel('Damage Classification', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=range(5), labels=["No Damage", "Affected", "Minor", "Major", "Destroyed"], rotation=30)
plt.show()

# Numerical correlation matrix chart

heatmap_numeric = heatmapdata.copy()

# Select meaningful numerical columns for correlation
correlation_columns = [
    'DamageScore', 
    'AssessedImprovedValue(parcel)', 
    'YearBuilt(parcel)',
    '#ofDamagedOutbuildings<120SQFT',
    '#ofNonDamagedOutbuildings<120SQFT'
]

# Calculate correlation matrix
corr_matrix = heatmap_numeric[correlation_columns].corr()

# Enhanced visualization
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True,  # Show correlation values
    fmt='.2f',   # Format to 2 decimal places
    cmap='coolwarm',  # Diverging color map
    center=0,    # Center color at 0
    square=True, # Make plot square
    linewidths=0.5,  # Add lines between cells
    cbar_kws={"shrink": .8},  # Slightly smaller color bar
    vmin=-1, vmax=1  # Ensure full correlation range
)
plt.title('Feature Correlation Heatmap\nFire Damage and Property Characteristics', fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:





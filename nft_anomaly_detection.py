import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from keras.models import Model
from keras.layers import Input, Dense

# Step 1: Load the dataset
data = pd.read_csv('nft_sales.csv')

# Step 2: Data Cleaning
# Remove dollar signs and commas, convert Sales to float
data['Sales'] = data['Sales'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Ensure Txns, Buyers, and Owners are numeric
data['Txns'] = pd.to_numeric(data['Txns'], errors='coerce')
data['Buyers'] = pd.to_numeric(data['Buyers'], errors='coerce')
data['Owners'] = pd.to_numeric(data['Owners'], errors='coerce')

# Check for missing values and data types
print("Missing values in each column:")
print(data.isnull().sum())
print("\nData types:")
print(data.dtypes)

# Handle missing values: Fill missing values only in numeric columns with the mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Step 3: Feature Engineering
data['Average Transaction Value'] = data['Sales'] / data['Txns']
data['Transaction Frequency'] = data['Txns'] / data['Buyers']
data['Owner-to-Buyer Ratio'] = data['Owners'] / data['Buyers']

# Step 4: Normalize the data
features = data[['Sales', 'Buyers', 'Txns', 'Owners', 
                  'Average Transaction Value', 'Transaction Frequency', 'Owner-to-Buyer Ratio']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Anomaly Detection using Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(scaled_features)

# Predict anomalies
data['isolation_anomaly'] = model.predict(scaled_features)

# Anomalies are labeled as -1
print("\nIsolation Forest Anomalies:")
print(data[data['isolation_anomaly'] == -1])

# Step 6: Anomaly Detection using Autoencoder
# Build the autoencoder model
input_dim = scaled_features.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation='relu')(input_layer)
encoder = Dense(16, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, validation_split=0.2)

# Predict and calculate reconstruction error
reconstructed = autoencoder.predict(scaled_features)
mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)

# Set a threshold for anomalies (using the 95th percentile)
threshold = np.percentile(mse, 95)
data['autoencoder_anomaly'] = mse > threshold

print("\nAutoencoder Anomalies:")
print(data[data['autoencoder_anomaly'] == True])

# Step 7: Visualize the Anomalies with Additional Graphs
plt.figure(figsize=(20, 15))

plt.figure(figsize=(20, 15))

# 1. Sales vs. Avg Txn Value (Isolation)
plt.subplot(3, 2, 1)
sns.scatterplot(x='Sales', y='Average Transaction Value', hue='isolation_anomaly', 
                data=data, palette={1: 'blue', -1: 'red'}, legend='full')
plt.title('Isolation Forest: Sales vs. Avg Transaction Value')
plt.xlabel('Sales'); plt.ylabel('Avg Transaction Value')

# 2. Owners vs. Buyers (Autoencoder)
plt.subplot(3, 2, 2)
sns.scatterplot(x='Owners', y='Buyers', hue='autoencoder_anomaly', 
                data=data, palette={True: 'red', False: 'blue'}, legend='full')
plt.title('Autoencoder: Owners vs. Buyers')
plt.xlabel('Owners'); plt.ylabel('Buyers')

# 3. Txn Freq vs. Buyers (Isolation)
plt.subplot(3, 2, 3)
sns.scatterplot(x='Transaction Frequency', y='Buyers', hue='isolation_anomaly', 
                data=data, palette={1: 'blue', -1: 'red'}, legend='full')
plt.title('Isolation Forest: Txn Freq vs. Buyers')
plt.xlabel('Transaction Frequency'); plt.ylabel('Buyers')

# 4. Owner-to-Buyer Ratio vs. Sales (Autoencoder)
plt.subplot(3, 2, 4)
sns.scatterplot(x='Owner-to-Buyer Ratio', y='Sales', hue='autoencoder_anomaly', 
                data=data, palette={True: 'red', False: 'blue'}, legend='full')
plt.title('Autoencoder: Owner-to-Buyer Ratio vs. Sales')
plt.xlabel('Owner-to-Buyer Ratio'); plt.ylabel('Sales')

# 5. Sales Distribution (Isolation)
plt.subplot(3, 2, 5)
sns.histplot(data=data, x='Sales', hue='isolation_anomaly', 
             palette={1: 'blue', -1: 'red'}, kde=True, bins=30)
plt.title('Isolation Forest: Sales Distribution')
plt.xlabel('Sales'); plt.ylabel('Frequency')

# 6. Buyers Distribution (Autoencoder)
plt.subplot(3, 2, 6)
sns.histplot(data=data, x='Buyers', hue='autoencoder_anomaly', 
             palette={True: 'red', False: 'blue'}, kde=True, bins=30)
plt.title('Autoencoder: Buyers Distribution')
plt.xlabel('Buyers'); plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


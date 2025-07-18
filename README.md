# NFT-Anomaly-Detection using Isolation Forest and Autoencoders

### ✅ Recommended File Structure

```
nft-anomaly-detection/
├── nft_anomaly_detection.py       # Main Python script (from your code)
├── nft_sales.csv                  # NFT dataset (input)
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

If you plan to modularize later, you could also create folders like `/utils` or `/plots`.

---

### 📄 `README.md`

```markdown
# 📉 NFT Anomaly Detection using Isolation Forest and Autoencoder

This project performs anomaly detection on NFT sales data using machine learning techniques like **Isolation Forest** and **Autoencoders**. It identifies unusual sales behavior by analyzing transactional features such as number of buyers, owners, sales volume, etc.

---

## 🧠 Techniques Used

- Data Cleaning and Feature Engineering
- Isolation Forest (unsupervised anomaly detection)
- Autoencoder (deep learning-based anomaly detection)
- Visualization with Matplotlib and Seaborn

---

## 📁 Project Structure

```

nft-anomaly-detection/
├── nft\_anomaly\_detection.py       # Main code for preprocessing, modeling, and visualization
├── nft\_sales.csv                  # Input dataset
├── requirements.txt               # Required Python libraries
└── README.md                      # You're reading this

````

---

## 📦 Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
````

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nft-anomaly-detection.git
   cd nft-anomaly-detection
   ```

2. Place your dataset in the same directory as `nft_anomaly_detection.py`, named:

   ```
   nft_sales.csv
   ```

3. Run the script:

   ```bash
   python nft_anomaly_detection.py
   ```

This will:

* Clean and preprocess the data
* Train both Isolation Forest and Autoencoder
* Predict and print anomalies
* Show anomaly visualizations

---

## 📊 Output

* **Prints tables** of anomalous entries flagged by both models.
* **Displays plots** for:

  * Anomaly distributions
  * Scatter plots of important features
  * Histograms of suspicious activity

---

## 📈 Features Used for Detection

* Sales Volume
* Number of Transactions (Txns)
* Number of Buyers
* Number of Owners
* Average Transaction Value
* Transaction Frequency
* Owner-to-Buyer Ratio




---

## 🧑‍💻 Author

* **pseudojan**
* Project: NFT Market Anomaly Detection – 2025

---








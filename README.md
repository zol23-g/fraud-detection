
# Fraud Detection Project

This repository contains the code and data for a fraud detection project. The project aims to identify fraudulent activities in e-commerce transactions and bank transactions using various machine learning models.

## Project Structure

```
fraud-detection/
│
├── data/
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│   └── model_explainability.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_explainability.py
│   ├── api/
│   │   ├── serve_model.py
│   │   └── requirements.txt
│   └── dashboard/
│       ├── app.py
│       └── requirements.txt
│
├── Dockerfile
├── README.md
└── .gitignore
```

## Datasets

### 1. Fraud_Data.csv
Includes e-commerce transaction data aimed at identifying fraudulent activities.
- `user_id`: A unique identifier for the user who made the transaction.
- `signup_time`: The timestamp when the user signed up.
- `purchase_time`: The timestamp when the purchase was made.
- `purchase_value`: The value of the purchase in dollars.
- `device_id`: A unique identifier for the device used to make the transaction.
- `source`: The source through which the user came to the site (e.g., SEO, Ads).
- `browser`: The browser used to make the transaction (e.g., Chrome, Safari).
- `sex`: The gender of the user (M for male, F for female).
- `age`: The age of the user.
- `ip_address`: The IP address from which the transaction was made.
- `class`: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.

### 2. IpAddress_to_Country.csv
Maps IP addresses to countries.
- `lower_bound_ip_address`: The lower bound of the IP address range.
- `upper_bound_ip_address`: The upper bound of the IP address range.
- `country`: The country corresponding to the IP address range.

### 3. creditcard.csv
Contains bank transaction data specifically curated for fraud detection analysis.
- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: These are anonymized features resulting from a PCA transformation.
- `Amount`: The transaction amount in dollars.
- `Class`: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.

## Tasks

### Task 1 - Data Analysis and Preprocessing
1. **Handle Missing Values**
   - Impute or drop missing values.
2. **Data Cleaning**
   - Remove duplicates.
   - Correct data types.
3. **Exploratory Data Analysis (EDA)**
   - Univariate analysis.
   - Bivariate analysis.
4. **Merge Datasets for Geolocation Analysis**
   - Convert IP addresses to integer format.
   - Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`.
5. **Feature Engineering**
   - Transaction frequency and velocity for `Fraud_Data.csv`.
   - Time-Based features for `Fraud_Data.csv`:
     - `hour_of_day`
     - `day_of_week`
6. **Normalization and Scaling**
7. **Encode Categorical Features**

### Task 2 - Model Building and Training
1. **Data Preparation**
   - Feature and Target Separation (`Class` for `creditcard`, `class` for `Fraud_Data`).
   - Train-Test Split.
2. **Model Selection**
   - Use several models to compare performance, including:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - Multi-Layer Perceptron (MLP)
     - Convolutional Neural Network (CNN)
     - Recurrent Neural Network (RNN)
     - Long Short-Term Memory (LSTM)
3. **Model Training and Evaluation**
   - Training models for both credit card and fraud-data datasets.
4. **MLOps Steps**
   - Versioning and Experiment Tracking:
     - Use tools like MLflow to track experiments, log parameters, metrics, and version models.

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages are listed in `src/api/requirements.txt` and `src/dashboard/requirements.txt`.

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r src/api/requirements.txt
   pip install -r src/dashboard/requirements.txt
   ```

### Usage

1. **Data Analysis and Preprocessing**:
   - Use the Jupyter notebooks in the `notebooks/` directory to perform data analysis and preprocessing.

2. **Model Training**:
   - Use the scripts in the `src/` directory to train and evaluate models.

3. **Serving the Model**:
   - Use the `src/api/serve_model.py` script to serve the trained model via an API.

4. **Dashboard**:
   - Use the `src/dashboard/app.py` script to run a dashboard for visualizing the results.

### Docker

To build and run the Docker container:
```sh
docker build -t fraud-detection .
docker run -p 5000:5000 fraud-detection
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

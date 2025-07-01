# Credit Card Fraud Detection Using Machine Learning

## 🧠 About the Project

Every day, thousands of credit card transactions are made—some legitimate, others fraudulent. This project focuses on using machine learning to detect those fraudulent transactions in real-time. It walks through building a classification model, handling imbalanced data, and evaluating model performance.

The goal? To help financial institutions flag suspicious transactions more effectively.

---

## 📊 Dataset Used

* **Name:** Credit Card Fraud Detection Dataset
* **Source:** [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Download Link:** [Download creditcard.csv](https://www.kaggle.com/mlg-ulb/creditcardfraud/download)
* **Total Records:** \~284,000
* **Features:**

  * Time, Amount
  * V1 to V28 (anonymized PCA components)
  * **Class** (Target): 0 = Genuine, 1 = Fraudulent

Note: The dataset is highly imbalanced—fraud cases make up only a small fraction of the data.

---

## ⚙️ What This Project Covers

| Step               | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| 🧹 Preprocessing   | Scaling features and splitting data into training/testing sets  |
| ⚖️ Class Imbalance | Applied SMOTE to balance fraud vs. non-fraud data               |
| 🤖 Models          | Trained Logistic Regression and Random Forest models            |
| 📈 Evaluation      | Used precision, recall, F1-score, AUC-ROC to assess performance |
| 📊 Visualizations  | ROC curves, confusion matrices, and feature importance charts   |

---

## 🔍 Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

These metrics give a well-rounded view of how well the models distinguish between fraud and non-fraud cases.

---

## 📁 Project Structure

```
fraud-detection-project/
├── creditcard.csv             # Dataset (download manually)
├── fraud_detection.py         # Python script with full pipeline
├── README.md                  # You're reading it!
```

---

## 🚀 How to Get Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fraud-detection-project.git
cd fraud-detection-project
```

### 2. Install the Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project folder.

### 4. Run the Script

```bash
python fraud_detection.py
```

You’ll see step-by-step logs, visual outputs, and performance metrics in the console.

---

## 🛠️ Tech Stack

* Python 3.7+
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* imbalanced-learn

To generate your own `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## 📌 What Can Be Improved Next?

* Save the trained model for deployment (using `joblib` or `pickle`)
* Add a Flask or Streamlit interface for real-time predictions
* Introduce more complex models like XGBoost or LightGBM
* Use MLflow or W\&B to track experiments

---

## 👤 About the Author

Hi, I'm **Rajvardhan Wakharde**, a B.Tech CSE student and data science enthusiast. This project is part of my hands-on learning in applied machine learning and cybersecurity. Always happy to connect and collaborate on similar projects!

Feel free to fork, star ⭐, or suggest improvements.

---

## 📄 License

This project is open-source and available under the MIT License.

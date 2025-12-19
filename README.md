# 🤖 ANN-Based Customer Churn Classification

A Machine Learning project that uses an **Artificial Neural Network (ANN)** to predict customer churn.  
This system helps businesses identify which customers are likely to leave the service, enabling proactive retention strategies.

---

## 🚀 Project Overview

Customer churn is when users stop using a company’s product or service. Predicting churn early is extremely valuable for businesses, especially in subscription-based or high-competition industries.

This project uses a neural network to learn patterns in customer data and classify whether a customer is likely to churn or not, based on multiple features such as demographics, usage metrics, and account information.

Whether you’re building data products or preparing for interviews, this churn classifier demonstrates practical application of deep learning for real-world problems.

---

## ✨ Key Features

- 🤖 **Artificial Neural Network model** for classification  
- 📊 Handles categorical and numerical data  
- 🧠 Trains and evaluates model performance  
- 📈 Includes accuracy and confusion matrix analysis  
- 🔄 Configurable preprocessing and scaling

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Main programming language |
| **TensorFlow / Keras** | ANN model training |
| **NumPy** | Numerical operations |
| **Pandas** | Data loading and manipulation |
| **Matplotlib / Seaborn** | Performance visualization |

---
## 📂 Project Structure

ANN-classification-churn/
│
├── data/
│ └── churn_dataset.csv
│
├── notebooks/
│ └── Churn_Analysis.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── train_model.py
│ └── evaluate.py
│
├── saved_model/
│ └── ann_churn_model.h5
│
├── requirements.txt
└── README.md


---

## ⚙️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/TrupalDholariya/ANN-classification-churn.git

2️⃣ Navigate to Project Folder
cd ANN-classification-churn

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ How to Run
✅ Data Preprocessing
python src/preprocess.py
```
## 🧠 Model Training

The neural network is trained using preprocessed customer data to learn meaningful patterns related to churn behavior.  
During training, the model adjusts its internal weights through backpropagation to minimize classification error.

Once training is complete, the finalized ANN model is saved locally so it can be reused for evaluation and future predictions without retraining.

---

## 📊 Model Evaluation

After training, the model is evaluated using unseen validation data to measure its performance.  
Key classification metrics such as accuracy, precision, recall, and F1-score are calculated to assess how well the model distinguishes between churned and non-churned customers.

A confusion matrix is also generated to provide a clear breakdown of correct and incorrect predictions.

---

## 🧠 Model Architecture Explained

The churn prediction model is built using a **feedforward Artificial Neural Network (ANN)** implemented with Keras.

The network processes normalized customer features through multiple hidden layers with non-linear activation functions.  
The final output layer performs binary classification, predicting whether a customer is likely to churn or remain active.

This architecture enables the model to generalize patterns from historical customer behavior and make reliable predictions.

---

## 🧪 Sample Results

After evaluation, the model produces performance metrics similar to the following:


Accuracy: 81.25%
Precision: 78.95%
Recall: 75.00%
F1-Score: 76.95%


These results indicate the model’s effectiveness in identifying potential churn customers while maintaining balanced performance across metrics.

---

## 🔮 Future Improvements

The project can be further enhanced in several ways to improve accuracy and usability, such as:

- Advanced hyperparameter tuning techniques
- Cross-validation for more robust evaluation
- Deeper or alternative neural network architectures
- Interactive dashboards for business insights
- Deployment as a web application or API

---

## 🤝 Contributing

Contributions are always welcome.  
If you would like to improve this project, feel free to fork the repository, implement enhancements, and submit a pull request.

Your contributions help make this project more impactful and production-ready.

---

## 📝 License

This project is distributed under the **MIT License**, allowing free use, modification, and distribution with appropriate credit.

---

## 🙌 Acknowledgements

This project is made possible thanks to the open-source machine learning community, high-quality educational resources, and the powerful tools provided by TensorFlow, Keras, and Python’s data science ecosystem.



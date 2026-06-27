# Chronic Kidney Disease Detection using Advanced Machine Learning Algorithms

This repository contains machine learning models and deployment tools to predict Chronic Kidney Disease (CKD) based on clinical data. The project explores a range of algorithms — from traditional classifiers to neural networks — to improve diagnostic accuracy, and ships a practical Streamlit web app for real-time predictions.

## Project Overview

Chronic Kidney Disease (CKD) is a progressive health issue, and early diagnosis is crucial for effective management. This project explores traditional and advanced machine learning approaches to classify CKD status using patient health records, focusing on a data-driven solution to aid in early detection and intervention.

## Project Structure

| File | Description |
| --- | --- |
| `kidney_disease.csv` | The dataset — 400 patient records with 25 clinical attributes plus the CKD classification label. |
| `predicting_chronic_kidney_disease.ipynb` | Main exploratory notebook: data cleaning, preprocessing, dimensionality reduction (PCA/LDA) and a comparison of several classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes and a Voting Classifier). |
| `Neural_Networks_for_CKD_Prediction.ipynb` | Builds and trains a Keras neural network for CKD classification, including data preprocessing, model architecture (with dropout to reduce overfitting), and training/evaluation over multiple epochs. |
| `Pycaret_to_predict_Kidney_diseases.ipynb` | A low-code modelling pass using PyCaret, which is very simple to implement for rapid model comparison. |
| `Model_Deployment.ipynb` | Trains a compact Random Forest on four key features (`age`, `wc`, `htn`, `dm`), saves it as `classifier.pkl`, and generates the Streamlit app (`app.py`). |
| `app.py` | Streamlit web application that loads the trained model and serves real-time CKD predictions. |
| `Problem_Statement_Prediction_of_Chronic_Kidney_Disease_using_Machine_Learning.pdf` | Problem statement, dataset details, and primary objectives. |
| `Chronic_kidney_disease_Information.pdf` | Medical background on CKD — causes, symptoms, and prevention strategies. |
| `requirements.txt` | Python dependencies required to run the notebooks and the app. |
| `LICENSE` | Project license. |

## Dataset and Attributes

**Name:** `kidney_disease.csv`

The dataset comprises 25 health indicators for 400 patients, including age, blood pressure, hemoglobin, and white blood cell count, along with the CKD classification label. Each attribute is carefully preprocessed (missing-value imputation, type fixing, encoding, scaling) before being used to train the models.

## Tech Stack

- **Language:** Python 3.8+
- **Data / ML:** NumPy, pandas, scikit-learn, XGBoost, imbalanced-learn
- **Deep learning:** TensorFlow / Keras
- **Visualisation:** Matplotlib, Seaborn, Plotly, Lux
- **Deployment:** Streamlit

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VinayakMokashi/KidneyDiseaseDetection-AdvancedML.git
cd KidneyDiseaseDetection-AdvancedML
```

### 2. Create a virtual environment (recommended) and install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Explore the notebooks

```bash
jupyter notebook
```

Open any of the `.ipynb` files to walk through the data exploration, preprocessing, model training and evaluation steps.

### 4. Run the Streamlit web app

The app needs the trained model file `classifier.pkl`. Generate it first by running **`Model_Deployment.ipynb`** end-to-end (it saves `classifier.pkl` in the project root), then launch the app:

```bash
streamlit run app.py
```

This opens an interactive page in your browser where you can enter a patient's hypertension status, diabetes status, age, and white blood cell count to get a CKD prediction.

## Project Workflow

1. **Exploration and Data Preprocessing** — Data is explored to identify missing values and anomalies, followed by preprocessing steps such as imputation, type fixing, encoding and normalization.
2. **Model Training** — Multiple classifiers (neural networks, random forest, extra trees, and several other statistical ML models) are trained to determine the most accurate model.
3. **Model Evaluation** — Performance metrics such as accuracy, precision, recall and F1-score assess each model's effectiveness, with the neural network and tree-based models showing promising results.
4. **Deployment** — A final model is deployed using Streamlit, creating an interactive tool for real-time CKD predictions.

## Future Work

- **Hyperparameter Tuning** — Enhance model performance with optimized parameters.
- **Larger Dataset** — Incorporate additional data to improve model robustness.
- **Enhanced Deployment** — Add features for a more user-friendly experience in the Streamlit app, and expand the deployed model beyond the current four features (`age`, `wc`, `htn`, `dm`) for stronger predictions.

## Disclaimer

This project is for educational and research purposes only. It is **not** a medical device and must not be used as a substitute for professional medical diagnosis or advice.

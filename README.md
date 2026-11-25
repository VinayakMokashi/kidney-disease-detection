# Chronic Kidney Disease Detection using Advanced Machine Learning Algorithms

This repository contains machine learning models and deployment tools to predict Chronic Kidney Disease (CKD) based on clinical data. The project explores a range of algorithms — from traditional classifiers to neural networks — under a leakage-free evaluation, and ships a practical Streamlit web app for real-time predictions.

## Project Overview

Chronic Kidney Disease (CKD) is a progressive health issue, and early diagnosis is crucial for effective management. This project explores traditional and advanced machine learning approaches to classify CKD status using patient health records, focusing on a data-driven solution to aid in early detection and intervention.

## Results

All numbers come from a split-first protocol (see [Evaluation protocol](#evaluation-protocol)): a stratified 80/20 split with every preprocessing step fitted on the training rows only, then a held-out test set of **80 patients** (50 CKD, 30 not CKD) and repeated stratified 5-fold cross-validation on all 400 patients.

| Model | Test accuracy (80 patients) | Cross-validated accuracy (mean ± sd) |
|:--|:--:|:--:|
| Random forest (entropy) + PCA | 100% | 99.9% ± 0.2% (5x5 CV) |
| Random forest (Gini) + PCA | 100% | 99.9% ± 0.5% (5x5 CV) |
| Decision tree + PCA | 98.8% | 99.4% ± 0.9% (5x5 CV) |
| **Neural network (Keras, 15-15-1 with dropout) + PCA** | **98.8%** | **99.3% ± 1.1%** (5x3 CV) |
| Gaussian naive Bayes + PCA | 97.5% | 99.0% ± 1.1% (5x5 CV) |
| KNN (k=3) + PCA | 100% | 98.5% ± 1.2% (5x5 CV) |
| SVM (RBF) + PCA | 97.5% | 98.3% ± 1.3% (5x5 CV) |
| Logistic regression + PCA | 97.5% | 97.2% ± 1.5% (5x5 CV) |
| Bernoulli naive Bayes + PCA | 92.5% | 92.2% ± 2.6% (5x5 CV) |
| PyCaret, best model (random forest) | 100% (114-row hold-out) | 99.6% (10-fold CV) |
| Streamlit app model: random forest on 4 inputs (`age`, `wc`, `htn`, `dm`) | 82.5% | 85.3% ± 3.8% (5x5 CV) |

The neural network's held-out test results: precision 1.00, recall 0.98, F1 0.99, ROC AUC 1.00 (one CKD patient missed).

This dataset is easy. Haemoglobin, packed cell volume, specific gravity, serum creatinine and albumin separate the classes almost on their own, so most full-feature models land between 97% and 100%. With only 80 test patients, one mistake moves test accuracy by 1.25 points, which is why the cross-validated column is the more reliable comparison. The app's four inputs were picked for ease of entry, not predictive power, and score lower.

## Evaluation protocol

1. **Split first.** A stratified 80/20 train/test split (`random_state=7`) is made before anything is fitted.
2. **Fit preprocessing on training rows only**, inside an `imblearn` pipeline: median / most-frequent imputation, one-hot encoding, random oversampling of the minority class, scaling to [-1, 1], and PCA (95% of variance) or LDA.
3. **Neural network:** a further validation split (64 rows) drives early stopping and the choice of decision threshold; the test set is used once.
4. **Cross-validation** repeats the whole pipeline, so oversampling and PCA are refitted inside every fold.

Label convention: 1 = CKD, 0 = not CKD.

## Project Structure

| File | Description |
| --- | --- |
| `kidney_disease.csv` | The dataset — 400 patient records with 24 clinical attributes, an `id` column and the CKD classification label. |
| `predicting_chronic_kidney_disease.ipynb` | Main notebook: data cleaning, exploratory analysis on the training split, and a comparison of nine classifiers (logistic regression, decision tree, two random forests, SVM, KNN, Gaussian and Bernoulli naive Bayes, and a voting ensemble) with PCA, LDA or no reduction, plus a tuned random forest. |
| `Neural_Networks_for_CKD_Prediction.ipynb` | Keras neural network (two hidden layers with dropout): train/validation/test split, early stopping, threshold chosen on validation, and repeated cross-validation. |
| `Pycaret_to_predict_Kidney_diseases.ipynb` | Low-code cross-check with PyCaret: model comparison, LDA tuning, and evaluation on hold-out and unseen rows. PyCaret 3.3 needs Python 3.9–3.11. |
| `Model_Deployment.ipynb` | Evaluates a compact random forest on four easy-to-enter features (`age`, `wc`, `htn`, `dm`), refits it on all rows, saves it as `classifier.pkl`, and generates the Streamlit app (`app.py`). |
| `app.py` | Streamlit web application that loads the trained model and serves real-time CKD predictions. |
| `Problem_Statement_Prediction_of_Chronic_Kidney_Disease_using_Machine_Learning.pdf` | Problem statement, dataset details, and primary objectives. |
| `Chronic_kidney_disease_Information.pdf` | Medical background on CKD — causes, symptoms, and prevention strategies. |
| `requirements.txt` | Python dependencies required to run the notebooks and the app. |
| `LICENSE` | Project license. |

## Dataset and Attributes

**Name:** `kidney_disease.csv` (UCI Chronic Kidney Disease dataset)

The dataset comprises 24 health indicators for 400 patients, including age, blood pressure, haemoglobin and white blood cell count, along with the CKD classification label (250 CKD, 150 not CKD). 242 records have at least one missing value. Some cells contain stray tabs or spaces (`'ckd\t'`, `'\tno'`, `'\t?'`), which the notebooks clean before modelling.

## Tech Stack

- **Language:** Python 3.10+
- **Data / ML:** NumPy, pandas, scikit-learn, imbalanced-learn
- **Deep learning:** TensorFlow / Keras
- **Visualisation:** Matplotlib, Seaborn
- **AutoML:** PyCaret (separate Python 3.9–3.11 environment)
- **Deployment:** Streamlit

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VinayakMokashi/kidney-disease-detection.git
cd kidney-disease-detection
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

The PyCaret notebook needs its own environment on Python 3.9–3.11:

```bash
python3.11 -m venv venv-pycaret
venv-pycaret/bin/pip install pycaret jupyter   # Windows: venv-pycaret\Scripts\pip
```

### 3. Explore the notebooks

```bash
jupyter notebook
```

Open any of the `.ipynb` files to walk through the data cleaning, preprocessing, model training and evaluation steps.

### 4. Run the Streamlit web app

The app needs the trained model file `classifier.pkl`. Generate it first by running **`Model_Deployment.ipynb`** end-to-end (it saves `classifier.pkl` in the project root), then launch the app:

```bash
streamlit run app.py
```

This opens an interactive page in your browser where you can enter a patient's hypertension status, diabetes status, age, and white blood cell count to get a CKD prediction.

## Project Workflow

1. **Cleaning** — Stray whitespace is stripped, numeric columns stored as text are converted, and the `id` column is set aside.
2. **Split** — A stratified train/test split is made before any model or transformer is fitted.
3. **Exploration** — Distributions, outliers and correlations are examined on the training rows.
4. **Model Training** — Multiple classifiers (neural network, random forests, SVM, KNN, naive Bayes, logistic regression and a voting ensemble) are trained in pipelines that impute, encode, oversample, scale and reduce dimensionality using training data only.
5. **Model Evaluation** — Accuracy, precision, recall, F1 and ROC AUC on the held-out test set, plus repeated cross-validation for stable estimates.
6. **Deployment** — A compact model is deployed with Streamlit for real-time CKD predictions.

## Future Work

- **Larger, external dataset** — 400 patients from one source is small; validating on data from another hospital would show whether the near-perfect scores transfer.
- **Stronger app model** — Expand the deployed model beyond `age`, `wc`, `htn` and `dm`: haemoglobin, specific gravity, serum creatinine and albumin carry far more signal.
- **Calibration** — Check that predicted probabilities match observed CKD rates before using them as risk scores.

## Disclaimer

This project is for educational and research purposes only. It is **not** a medical device and must not be used as a substitute for professional medical diagnosis or advice.

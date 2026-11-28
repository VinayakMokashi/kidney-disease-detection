# Chronic Kidney Disease Detection

This repository contains machine learning models that predict Chronic Kidney Disease (CKD) from 24 clinical attributes in the UCI CKD dataset (400 patients), and a small Streamlit app that uses four of them. The project compares a range of algorithms — from traditional classifiers to a small neural network — under a leakage-free evaluation. The dataset is small and easy, so near-perfect scores are expected here; see [Results](#results).

## Project Overview

Chronic Kidney Disease (CKD) is a progressive health issue, and early diagnosis is crucial for effective management. This project classifies CKD status from patient health records with standard machine learning models and a small neural network.

The tasks come from a third-party assignment brief, included as [`Problem_Statement_Prediction_of_Chronic_Kidney_Disease_using_Machine_Learning.pdf`](Problem_Statement_Prediction_of_Chronic_Kidney_Disease_using_Machine_Learning.pdf): handle missing values, select features, compare up to 11 classifiers and report the most accurate one. The brief does not name its author or course. Part of its introduction is the abstract of I. U. Ekanayake and D. Herath, [*Chronic Kidney Disease Prediction Using Machine Learning Methods*](https://doi.org/10.1109/MERCon50084.2020.9185249) (MERCon 2020), a paper that compared 11 machine learning methods for CKD prediction. The notebooks use PCA or LDA instead of a separate feature-selection step, and do not use XGBoost, one of the brief's 11 models.

## Results

All numbers come from a split-first protocol (see [Evaluation protocol](#evaluation-protocol)): a stratified 80/20 split with every preprocessing step fitted on the training rows only, then a held-out test set of **80 patients** (50 CKD, 30 not CKD) and repeated stratified 5-fold cross-validation on all 400 patients. The cross-validation folds include the 80 test patients, so the two columns are not independent.

| Model | Test accuracy (80 patients) | Cross-validated accuracy (mean ± sd) |
|:--|:--:|:--:|
| Random forest (entropy) + PCA | 100% (80/80) | 99.95% ± 0.24% (5x5 CV) |
| Random forest (Gini) + PCA | 100% (80/80) | 99.90% ± 0.49% (5x5 CV) |
| Decision tree + PCA | 98.75% (79/80) | 99.40% ± 0.94% (5x5 CV) |
| Neural network (Keras, 15-15-1 with dropout) + PCA, threshold 0.5 | 98.75% (79/80) | 99.25% ± 1.14% (5x3 CV) |
| Gaussian naive Bayes + PCA | 97.5% (78/80) | 99.00% ± 1.12% (5x5 CV) |
| KNN (k=3) + PCA | 100% (80/80) | 98.55% ± 1.16% (5x5 CV) |
| SVM (RBF) + PCA | 97.5% (78/80) | 98.25% ± 1.32% (5x5 CV) |
| Logistic regression + PCA | 97.5% (78/80) | 97.20% ± 1.47% (5x5 CV) |
| Bernoulli naive Bayes + PCA | 92.5% (74/80) | 92.20% ± 2.58% (5x5 CV) |
| PyCaret, best model (random forest) | 100% (114/114 on PyCaret's own hold-out) | 99.63% (10-fold CV) |
| Streamlit app model: random forest on 4 inputs (`age`, `wc`, `htn`, `dm`) | 82.5% (66/80) | 85.25% ± 3.77% (5x5 CV) |

The neural network's held-out test results at the default 0.5 threshold: precision 1.00, recall 0.98, F1 0.99, ROC AUC 1.00 (one CKD patient missed). At the threshold chosen on the validation rows (0.890), its test accuracy is 97.5% (78/80).

This dataset is easy. Haemoglobin, packed cell volume, specific gravity, serum creatinine and albumin separate the classes almost on their own, so most full-feature models land between 97% and 100%. Those scores say more about this 400-patient dataset than about the models, and there is no evidence yet that they would hold on patients from another source. With only 80 test patients, one mistake moves test accuracy by 1.25 points, which is why the cross-validated column is the more reliable comparison. The app's four inputs were picked for ease of entry, not predictive power, and score lower.

## Evaluation protocol

1. **Split first.** A stratified 80/20 train/test split (`random_state=7`) is made before anything is fitted.
2. **Fit preprocessing on training rows only**, inside an `imblearn` pipeline: median / most-frequent imputation, one-hot encoding, random oversampling of the minority class, scaling to [-1, 1], and PCA (95% of variance) or LDA.
3. **Neural network:** a further validation split (64 rows) drives early stopping and the choice of a decision threshold. The test set is scored twice, at the default 0.5 threshold and at the validation-chosen one; both results are reported above.
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
| `app.py` | Streamlit form that loads the 4-feature model (`classifier.pkl`) and returns a CKD prediction. |
| `Problem_Statement_Prediction_of_Chronic_Kidney_Disease_using_Machine_Learning.pdf` | The third-party assignment brief this project follows: problem statement, dataset details and tasks. |
| `Chronic_kidney_disease_Information.pdf` | Medical background on CKD — causes, symptoms, and prevention strategies — from the same source as the brief. |
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
- **App:** Streamlit

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

This opens an interactive page in your browser where you can enter a patient's hypertension status, diabetes status, age, and white blood cell count to get a CKD prediction. The app's model is the weakest in the results table: 82.5% test accuracy, and it misses 14 of the 50 CKD patients in the test set.

## Project Workflow

1. **Cleaning** — Stray whitespace is stripped, numeric columns stored as text are converted, and the `id` column is set aside.
2. **Split** — A stratified train/test split is made before any model or transformer is fitted.
3. **Exploration** — Distributions, outliers and correlations are examined on the training rows.
4. **Model Training** — Multiple classifiers (neural network, random forests, SVM, KNN, naive Bayes, logistic regression and a voting ensemble) are trained in pipelines that impute, encode, oversample, scale and reduce dimensionality using training data only.
5. **Model Evaluation** — Accuracy, precision, recall, F1 and ROC AUC on the held-out test set, plus repeated cross-validation for stable estimates.
6. **Streamlit app** — A compact 4-feature model backs a Streamlit form that runs locally.

## Future Work

- **Larger, external dataset** — 400 patients from one source is small; validating on data from another hospital would show whether the near-perfect scores transfer.
- **Stronger app model** — Expand the app's model beyond `age`, `wc`, `htn` and `dm`: haemoglobin, specific gravity, serum creatinine and albumin carry far more signal.
- **Calibration** — Check that predicted probabilities match observed CKD rates before using them as risk scores.

## Disclaimer

This project is for educational and research purposes only. It is **not** a medical device and must not be used as a substitute for professional medical diagnosis or advice.

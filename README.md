# Heart Disease Prediction

Predicting heart disease using machine learning on clinical patient data.

This project builds a machine learning model to predict whether a patient has heart disease based on medical attributes from the UCI Cleveland dataset. Achieves high accuracy through exploratory data analysis, model comparison, and hyperparameter tuning with Scikit-Learn.

## Table of Contents

- [Heart Disease Prediction](#heart-disease-prediction)
  - [Table of Contents](#table-of-contents)
  - [Problem Definition](#problem-definition)
  - [Dataset](#dataset)
  - [Evaluation](#evaluation)
  - [Features](#features)
  - [Models](#models)
  - [Results](#results)
    - [Key Visualizations](#key-visualizations)
    - [Model Performance Metrics (Best Logistic Regression)](#model-performance-metrics-best-logistic-regression)
    - [Confusion Matrix](#confusion-matrix)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Key Techniques Used](#key-techniques-used)
  - [Next Steps](#next-steps)
  - [Dependencies](#dependencies)
  - [License](#license)

## Problem Definition

Given clinical parameters about a patient, predict whether they have heart disease.

## Dataset

[Original Cleveland dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) from UCI Machine Learning Repository (303 samples, 14 features).

Available on [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci).

## Evaluation

Target: 95% accuracy on test set for proof of concept.

## Features

| Feature    | Description                  |
|------------|------------------------------|
| `age`      | Age in years                 |
| `sex`      | Sex (1 = male, 0 = female)   |
| `cp`       | Chest pain type              |
| `trestbps` | Resting blood pressure       |
| `chol`     | Serum cholesterol (mg/dl)    |
| `fbs`      | Fasting blood sugar > 120    |
| `restecg`  | Resting ECG results          |
| `thalach`  | Maximum heart rate           |
| `exang`    | Exercise induced angina      |
| `oldpeak`  | ST depression                |
| `slope`    | Slope of peak exercise ST    |
| `ca`       | Number of major vessels      |
| `thal`     | Thalassemia                  |
| `target`   | Heart disease (0 = no, 1 = yes) |

## Models

| Model              | Base Accuracy | Tuned Accuracy |
|--------------------|---------------|----------------|
| Logistic Regression| 85%           | 88%            |
| K-Nearest Neighbors| 83%          | 86%            |
| Random Forest      | 82%           | 87%            |

**Best model**: Logistic Regression (88% accuracy, cross-validated precision/recall/F1 all >87%).

## Results

### Key Visualizations

- **Target distribution**: Balanced classes (165 disease, 138 no disease)
- **Age vs. Max Heart Rate**: Clear separation between diseased/non-diseased patients
- **Chest pain correlation**: Higher frequency of disease with certain pain types
- **Feature importance**: `cp`, `thalach`, `oldpeak` most predictive

### Model Performance Metrics (Best Logistic Regression)

```
Cross-Validation Scores:
- Accuracy: 88.07%
- Precision: 87.14%
- Recall: 88.52%
- F1 Score: 87.80%
```

### Confusion Matrix

```
[[25  4]
 [ 3 26]]
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the analysis notebook
jupyter notebook heart_disease_prediction.ipynb

# Load saved model for predictions
python predict.py
```

**Sample prediction**:

```python
import pickle
import pandas as pd

model = pickle.load(open("heart-disease-model.pkl", "rb"))
sample_patient = pd.read_csv("data/new_patient.csv")
prediction = model.predict(sample_patient)
print("Heart Disease Prediction:", "Yes" if prediction[0] == 1 else "No")
```

## Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart-disease.csv      # UCI Cleveland dataset
├── heart_disease_prediction.ipynb  # Main analysis notebook
├── heart-disease-model.pkl    # Trained model
├── requirements.txt           # Dependencies
├── predict.py                 # Prediction script
└── README.md                  # This file
```

## Key Techniques Used

1. **Exploratory Data Analysis**: Correlation matrices, outlier detection, cross-tabulations
2. **Visualization**: Seaborn heatmaps, Matplotlib scatter plots, histograms
3. **Model Comparison**: Logistic Regression, KNN, Random Forest
4. **Hyperparameter Tuning**: GridSearchCV across multiple models
5. **Evaluation**: Cross-validation, ROC curves, confusion matrices, classification reports
6. **Feature Engineering**: Normalized feature importance visualization

## Next Steps

- Ensemble methods (Stacking, Voting Classifier)
- Feature selection (RFE, SelectKBest)
- Advanced models (XGBoost, LightGBM)
- Model deployment (Flask/FastAPI API)
- Real-time patient monitoring integration

## Dependencies

See [requirements.txt](requirements.txt):

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## License

MIT License - see [LICENSE](LICENSE) for details.
```
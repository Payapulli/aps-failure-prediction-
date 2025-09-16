# APS Failure Prediction for Scania Trucks

A machine learning project that predicts Air Pressure System (APS) failures in Scania trucks using sensor data. This project demonstrates advanced ML techniques including Random Forest, XGBoost, and handling severe class imbalance.

## 🚛 Project Overview

This project addresses a real-world industrial problem: predicting APS (Air Pressure System) failures in Scania trucks to enable predictive maintenance and reduce downtime costs. The dataset contains sensor readings from truck components, and the goal is to classify whether a failure will occur.

## 📊 Dataset

- **Source**: Scania CV AB (Sweden)
- **Training samples**: 60,000
- **Test samples**: 16,000
- **Features**: 170 sensor readings
- **Target**: Binary classification (failure vs no failure)
- **Class distribution**: 98.3% negative, 1.7% positive (severely imbalanced)

## 🛠️ Technical Approach

### Data Preprocessing
- **Missing value imputation** using mean imputation
- **Feature analysis** including coefficient of variation
- **Data visualization** with correlation matrices and scatter plots

### Machine Learning Models
1. **Random Forest**
   - Out-of-bag error estimation
   - Class weight balancing for imbalanced data
   - Feature importance analysis

2. **XGBoost**
   - L1 regularization (Lasso penalty)
   - 5-fold cross-validation
   - Hyperparameter tuning

3. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Class weight adjustment
   - Performance trade-off analysis

### Model Evaluation
- **Confusion matrices** for detailed performance analysis
- **ROC curves and AUC** for classification performance
- **Cross-validation** to prevent overfitting
- **Misclassification rates** on training and test sets

## 📈 Results

### XGBoost Performance
- **Test Accuracy**: 99.34%
- **AUC Score**: 0.99
- **Cross-validation Error**: 0.58%

### Random Forest Performance
- **Test Accuracy**: 99.23%
- **AUC Score**: 0.99
- **Out-of-bag Error**: 0.41%

### Class Imbalance Impact
- **Without balancing**: High accuracy but poor minority class performance
- **With SMOTE/class weights**: Improved minority class recall at cost of overall accuracy
- **Business trade-off**: Better failure detection vs false alarms

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aps-failure-prediction.git
cd aps-failure-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
jupyter notebook notebooks/aps_failure_analysis.ipynb
```

## 📁 Project Structure

```
aps-failure-prediction/
├── data/
│   ├── aps_failure_training_set.csv
│   ├── aps_failure_test_set.csv
│   └── aps_failure_description.txt
├── notebooks/
│   └── aps_failure_analysis.ipynb
├── results/
│   └── (generated visualizations)
├── src/
│   └── (utility functions)
├── requirements.txt
└── README.md
```

## 🔍 Key Insights

1. **Feature Engineering**: Coefficient of variation analysis revealed the most variable features
2. **Class Imbalance**: Severe imbalance required specialized techniques (SMOTE, class weights)
3. **Model Performance**: XGBoost slightly outperformed Random Forest
4. **Business Value**: High accuracy enables reliable predictive maintenance

## 🎯 Business Impact

- **Predictive Maintenance**: Early failure detection reduces unplanned downtime
- **Cost Savings**: Prevents expensive emergency repairs
- **Operational Efficiency**: Optimizes maintenance scheduling
- **Safety**: Reduces risk of equipment failure during operation

## 📚 Technical Skills Demonstrated

- **Machine Learning**: Random Forest, XGBoost, Cross-validation
- **Data Preprocessing**: Missing value imputation, feature engineering
- **Class Imbalance**: SMOTE, class weights, performance analysis
- **Model Evaluation**: ROC curves, confusion matrices, AUC scores
- **Data Visualization**: Correlation matrices, scatter plots, box plots
- **Python Libraries**: scikit-learn, XGBoost, pandas, matplotlib, seaborn

## 📄 License

This project uses the Scania APS Failure dataset, which is licensed under the GNU General Public License v3.0.

## 👨‍💻 Author

**Joshua Payapulli**
- GitHub: [@Payapulli](https://github.com/Payapulli)
- USC ID: 3751786221

## 🙏 Acknowledgments

- Scania CV AB for providing the dataset
- University of Southern California for course materials
- The open-source ML community for excellent libraries

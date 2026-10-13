# CodeAlpha_IrisClassification

**CodeAlpha Data Science Internship - Task 1: Iris Flower Classification**

Classifies Iris flowers (setosa, versicolor, virginica) from sepal and petal measurements using scikit-learn.

## Approach
1. Load the Iris dataset (150 samples, 4 features, 3 classes) and check for missing values (none).
2. Explore the data with a pair plot and correlation heatmap.
3. Stratified 80/20 train/test split.
4. Compare five models with 5-fold cross-validation: Logistic Regression, KNN, SVM, Decision Tree, Random Forest.
5. Evaluate the best model on the test set (accuracy, precision/recall/F1, confusion matrix).

## Results
| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.9583 | 0.9333 |
| KNN | 0.9667 | 0.9333 |
| SVM | 0.9667 | **0.9667** |
| Decision Tree | 0.9417 | 0.9333 |
| Random Forest | 0.9500 | 0.9000 |

Best model: **SVM** with 96.67% test accuracy. Setosa is separated perfectly; the only confusion is between versicolor and virginica.

## Run
```
pip install -r requirements.txt
python iris_classification.py
```

## Files
- `iris_classification.py` - full pipeline
- `pairplot.png`, `correlation_heatmap.png`, `confusion_matrix.png` - plots
- `model_comparison.csv` - results table

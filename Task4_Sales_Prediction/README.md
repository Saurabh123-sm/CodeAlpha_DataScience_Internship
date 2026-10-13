# CodeAlpha_SalesPrediction

**CodeAlpha Data Science Internship - Task 4: Sales Prediction using Python**

Predicts product sales from advertising spend across three channels (TV, Radio, Newspaper) and turns the model into marketing insights.

## Approach
1. **Data cleaning:** 200 rows, no missing values, no duplicates.
2. **Exploration:** pair plot and correlation heatmap. Correlation with Sales: TV 0.78, Radio 0.58, Newspaper 0.23.
3. **Modelling:** 80/20 split and 5-fold CV on four regressors.
4. **Insight:** coefficients of a linear model plus feature importances from the best model.

## Results
| Model | CV R2 | Test R2 | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.8595 | 0.8994 | 1.461 | 1.782 |
| Ridge | 0.8595 | 0.8994 | 1.461 | 1.782 |
| Random Forest | 0.9701 | 0.9826 | 0.614 | 0.741 |
| **Gradient Boosting** | 0.9697 | **0.9831** | 0.619 | 0.730 |

Best model: **Gradient Boosting** (R2 = 0.983 on held-out data).

## Business insights
- **Radio gives the highest return per unit spent:** +1 unit of Radio spend adds about 0.19 units of sales, versus about 0.046 for TV (other channels held constant).
- **TV is still the largest overall driver** (feature importance 61%) because of its higher spend volume.
- **Newspaper has essentially no effect** once TV and Radio are accounted for (coefficient about 0, importance under 1%). Its apparent correlation with sales comes from being correlated with Radio spend. Newspaper budget is the best candidate to reallocate to Radio/TV.
- Tree models outperform linear ones, suggesting some non-linear effects/interaction (diminishing returns at high spend).

## Run
```
pip install -r requirements.txt
python sales_prediction.py
```

## Files
- `sales_prediction.py` - full pipeline
- `advertising.csv` - dataset
- `pairplot.png`, `correlation_heatmap.png`, `actual_vs_predicted.png`, `channel_impact.png` - plots
- `model_comparison.csv` - metrics table

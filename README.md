# Credit-Card-Default-Prediction-using-Deep-Learning
# Project Overview

Predicting credit card default is challenging due to imbalanced data and changing customer repayment behavior. This project develops a machine learning and deep learning model to identify high-risk customers who are likely to default.

Using a dataset of 30,000 credit card clients from Taiwan, the goal is to detect potential defaulters early so financial institutions can reduce financial risk and improve lending decisions.

# Dataset
The dataset includes information about:
1. Customer demographics
2. Credit limit
3. Bill statements
4. Payment history
5. Monthly repayment status
6. Target variable:
    default
    1 = Default
    0 = Non-Default
# Class distribution:
    78% Non-default
    22% Default
# Data Cleaning
Key preprocessing steps included:
1. Fixed column headers and corrected numeric data types
2. Removed unnecessary columns such as ID
3. Cleaned and standardized education categories
4. Renamed the target variable to default for clarity
# Feature Engineering
To better capture repayment behavior, several behavioral features were created:
1. pay_trend – Overall repayment improving or worsening over time
2. recent_vs_past_pay – Detects sudden recent payment delays
3. payment_variability – Measures instability in monthly payments
4. delay_acceleration – Speed at which repayment delays are increasing
5. longest_pay_delay_streak – Longest consecutive months of delayed payments
These engineered features help the model better capture repayment patterns and credit risk behavior.
# Modeling Strategy
Three models were developed and compared:
1. Baseline Model
   Initial model without handling class imbalance.
2. Class-Weighted Model
    To address the dataset imbalance:
    Non-default weight: 0.64
    Default weight: 2.25
This helped the model focus more on identifying high-risk customers.
3. Tuned Deep Learning Model
   Architecture:
   Dense(256)
   Batch Normalization
   Dropout

   Dense(128)
   Batch Normalization
   Dropout

   Dense(64)
   Dropout

   Output Layer (Sigmoid)
Regularization techniques used:
   Batch Normalization
   Dropout
These help reduce overfitting and improve generalization.
# Model Performance
1. Class Weighted Model Improvements
2. Recall improved: 0.32 -> 0.62
3. False Negatives reduced: 677 -> 378
4. F1 Score improved: 0.44 -> 0.53

# Final Tuned Model Performance (Threshold = 0.55)
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.80  |
| Precision | 0.56  |
| Recall    | 0.54  |
| F1 Score  | 0.55  |
Additional Results:
 -  599 defaulters correctly identified
 -  Improved balance between false alarms and missed defaulters
# Business Impact
This model helps financial institutions:
1. Identify high-risk customers early
2. Reduce potential credit losses
3. Improve credit risk management strategies
4. Support data-driven lending decisions

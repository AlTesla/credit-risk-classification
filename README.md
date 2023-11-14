# credit-risk-classification
## Overview of the Analysis 
The aim of this analysis is to use different methods to train and assess a model that is based on loan risk. I utilized a dataset of past lending activity from a peer-to-peer lending services firm to create a model that can recognize the creditworthiness of borrowers.

The financial data included the following information: the size of the loan, the interest rate, the borrower’s income, the debt-to-income ratio, the number of accounts the borrower held, derogatory marks against the borrower, and the total debt. The objective was to predict the loan status based on the available data. 

In the initial stage of the machine learning process, the data was divided into two sets using the “train_test_split” function of the scikit-learn library. One set was used for training the logistic regression model using the “LogisticRegression” function from the same library, while the other set was used to test the performance of the trained model in classifying the borrower as high or low risk.

In the second stage of the machine learning process, the “RandomOverSampler” function from the imbalanced-learn library was used to resample the training data. This was done to ensure that the logistic regression model had an equal number of data points to draw from, since the initial model was drawing from a dataset that had 75,036 low-risk loan data points and 2,500 high-risk data points. The resampling of the training data resulted in having 56,271 data points for both low and high risk loans.

## Results 
### Prior to using the “RandomOverSampler” function, the following results were obtained: 
- The logistic regression model was 95% accurate at predicting healthy vs high-risk loan labels before using the "RandomOverSampler" function.
- According to the classification report, the model was pretty accurate at predicting healthy instances, with a precision of 1.00, a recall of 0.99, and an f1-score of 1.00.
- However, the model was not very accurate at predicting high-risk instances, with a precision of 0.85, a recall of 0.91, and an f1-score of 0.88.
### After applying the “RandomOverSampler” function, the following results were obtained: 
- The logistic regression model was 99% accurate at predicting healthy vs high-risk loan labels according to the balanced_accuracy score.
- According to the classification report, the model was pretty accurate at predicting healthy instances, with a precision of 1.00, a recall of 0.99, and an f1-score of 1.00.
- However, the model was not very accurate at predicting high-risk instances, with a precision of 0.84, a recall of 0.99, and an f1-score of 0.91.

##  Summary
The logistic regression model’s performance improved significantly after resampling the data, with its accuracy increasing from 95% to 99% according to the balanced_accuracy score. Both models performed well in predicting healthy loans, but for high-risk loans, resampling resulted in better recall and accuracy ratios, albeit with slightly lower precision. In summary, the second model had fewer false predictions overall and is better to be used based on its higher accuracy and recall.
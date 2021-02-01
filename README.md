# Risky Business

![Credit Risk](Images/credit-risk.jpg)
In this project, I constructed and evaluated several machine-learning techniques used to predict credit risk for peer-to-peer lending companies. Companies, such as LendingClub or Prosper, allow investors to make loans to other people without the use of banks. This type of lending can be high-risk without the right machine-learning model, so finding the correct algorithm will allow them to mitigate risk and incentivize more investments among their users.

Considering that credit risk is an imbalanced classification problem (the number of good loans is much higher than the number of at-risk loans), I employed different techniques for training and evaluating models with imbalanced classes. In the first technique, I resampled the data using four different algorithms from the imbalanced-learn library. I then used this resampled data to build a logistics regression classifier and evaluated the performance of each of the four models. 

In the second technique, I used the unsampled data to create and evaluate two ensemble classifier, a `Balanced Random Forest classifier`, and an `Easy Ensemble AdaBoost classifier`. 


## Resampling Technique Overview ([credit_risk_resampling.ipynb](https://github.com/EmilianoAmador/Unit_11_Classification_Risky_Business/blob/master/Code/credit_risk_resampling.ipynb))

Used the imbalanced-learn library to resample the quarterly data from LendingClub:

1. Oversampled the data using the `Naive Random Oversampler` and `SMOTE` algorithms.
2. Undersampled the data using the `Cluster Centroids` algorithm.
3. Over- and under-sampled using a combination `SMOTEENN` algorithm.

For each resampled data above:

1. Trained `logistic regression classifier` from `sklearn.linear_model`.
2. Calculated the `balanced accuracy score` from `sklearn.metrics`.
3. Calculated the `confusion matrix` from `sklearn.metrics`.
4. Printed the `imbalanced classification report` from `imblearn.metrics`.

## Results:

**Parameters set for Logistic Regression Model:**
<br/>
![Doc File](Images/Logistic_Reg_Parameters.png)

## Naive Random Oversampling

**Confusion Matrix:**
<br/>
Here, we can see how many times the model predicted high risk and low risk correctly as well as incorrectly. The first number, 76, is the True Positive (TP) it represents the number of times the model predicted high risk and turned out to be truly high risk. The number below that, 4493, is the True Negative (TN) and represents the number of times the model correctly predicted it was a low risk. The second column is the all the ones the model predicted wrong. The False Positive (FP), 25, is the number of times the model predicted high risk and turned out not to be a low risk; while the False Negative (FN), 12611, is the number the model predicted low risk but turned out to be high risk. 

<br/>
![Doc File](Images/NaiveOversamp_Matrix.png)
<br/>
<br/>
<br/>
**Imbalanced Calssification Report:**
<br/>
The classification report demonstrates a better overview of what the confusion matrix means in terms of efficiency. The precision (pre) is the percentage of individuals that the model correctly predicted to be high risk. In other words, it is the TP divided by the sum of TP and FP. The recall (rec), unlike the precision, demonstrates the percentage of individuals the model correctly predicted to be high risk and low risk. The f1-score is the harmonic mean between the precision and the recall. It represents the frequency at which the model predicts high risk or low risk correctly. 
<br/>
The last row of numbers represents the accuracy of the model's scores. Below it states that the model is 99 percent accurate at determining high risk; however, this number is deceiving when it comes to imbalanced classes because of the accuracy paradox. This paradox states that in a dataset of 1 high risk and 100 low risks individuals the model can get away with just predicting low risk all the time and receive a great accuracy score; however, when it guesses a high-risk wrong there can be costly repercussions. For this, we use the index of imbalanced accuracy (iba) to give us a more accurate representation of the accuracy for the imbalanced dataset and eliminates the influence from the dominating class (ie the 100 low-risk individuals).
<br/>
![](Images/NOS_Classification_Report.png)
<br/>
<br/>

## Smote Oversampling

**Confusion Matrix:**
<br/>
![Doc File](Images/Smote_matrix.png)
<br/>
<br/>
<br/>
**Imbalanced Calssification Report:**
<br/>
![](Images/Smote_CLassificiationReport.png)
<br/>
<br/>

## Cluster Centroids Undersampling

**Confusion Matrix:**
<br/>
![Doc File](Images/ClusterCentroidMatrix.png)
<br/>
<br/>
<br/>
**Imbalanced Calssification Report:**
<br/>
![](Images/Cluster_ClassificationReport.png)
<br/>
<br/>

## Smoteen Combination Sampling

**Confusion Matrix:**
<br/>
![Doc File](Images/SmoteenMatrix.png)
<br/>
<br/>
<br/>
**Imbalanced Calssification Report:**
<br/>
![](Images/Smoteen_ClassificationReport.png)
<br/>
<br/>

**Conclusion:**
<br/>
![](Images/Conclusion.png)
<br/>
The Naive Random Oversampling out-performs the rest of the sampling techniques. As seen on the table above, it has a better overall Accuracy, Recall, and Geometric Mean. This means that it is better at predicting credit risk. 

## Ensemble Learning Technique ([credit_risk_ensemble.ipynb](https://github.com/EmilianoAmador/Unit_11_Classification_Risky_Business/blob/master/Code/credit_risk_ensemble.ipynb))

I trained and compared two different ensemble classifiers to predict loan risk and evaluate each model. I used the `balanced random forest classifier` and the `easy ensemble AdaBoost classifier`, and used 100 estimators for each. 

For each ensemble classifier:

1. Trained the model using the quarterly data from LendingClub.
2. Calculated the balanced accuracy score from `sklearn.metrics`.
3. Printed the confusion matrix from `sklearn.metrics`.
4. Generated a classification report using the `imbalanced_classification_report` from imbalanced learn.

## Results:
## Balanced Random Forest

**Parameters set for the model:**
<br/>
![Doc File](Images/Blanced_Random_Forest-Parameters.png)
<br/>
<br/>
<br/>
**Confusion Matrix:**
<br/>
![Doc File](Images/RF_matrix.png)
<br/>
<br/>
<br/>
**Imbalanced Calssification Report:**
<br/>
![](Images/Balanced_Random_forest_CL-ClassificationReport.png)
<br/>
<br/>
## Easy Ensemble Ada Boost

**Parameters set for the model:**
<br/>
![Doc File](Images/EasyEnsemble-Parameters.png)
<br/>
<br/>
<br/>
**Confusion Matrix:**
<br/>
![Doc File](Images/Easy_matrix.png)
<br/>
<br/>
<br/>
**Imbalanced Calssification Report:**
<br/>
![Doc File](Images/Easy_ensemble-ClassificationReport.png)
<br/>
<br/>

#### Conclusions:

![Doc File](Images/Both_compared.png)
<br/>
<br/>
When it comes to imbalanced classes such as credit risk, it is important to use multiple models as well as multiple measures to assess their accuracy. No one model should be used blindly when assessing credit risk because these algorithms hardly take into account spontaneous factors such as world events that would influence a person's ability to pay back a loan. For example, a person could be low risk, but if a large sudden unemployment rate causes them to lose their source of income then the model predicted wrong. This is due to the model's inability to predict future sudden events. Therefore, investors need to use the models as support to make a decision but not as the sole decision maker.  Investors should find a weighted balance between a model's decision and their judgement from environmental factors that may prove the model wrong. 
<br/>
Below are factors that can help assess risk for investors in order of importance. 
<br/>
<br/>
![Doc File](Images/Features_table.png)

# Risky Business

![Credit Risk](Images/credit-risk.jpg)
In this project, I built and evaluated several machine-learning techniques that will be used to predict credit risk for peer-to-peer lending companies. These types of companies, such as LendingCLub or Prosper, allow investors make loans to other people without the use of banks. This type of lending can be very risky without the right machine-learning model, so finding the right algorithm will allow them mitigate risk and incentivise more investments.

Considering that credit risk is an imbalanced classification problem (the number of good loans is much higher than the number of at-risk loans), I employed different techniques for training and evaluating models with imbalanced classes. In the first technique, I resampled the data using four different algorithms from imbalanced-learn library. I then used this resampled data to build a logistics regression classifier and evaluated the performance of each of the four models. In the second technique, I used the unsampled data to create and evaluate two ensemble classifier, a `balanced random forest classifier` and an `easy ensemble AdaBoost classifier`. 


#### Resampling Technique

Used the imbalanced-learn library to resample the quarterly data from LendingClub:

1. Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms.
2. Undersample the data using the `Cluster Centroids` algorithm.
3. Over- and under-sample using a combination `SMOTEENN` algorithm.

For each resampled data above:

1. Trained `logistic regression classifier` from `sklearn.linear_model`.
2. Calculated the `balanced accuracy score` from `sklearn.metrics`.
3. Calculated the `confusion matrix` from `sklearn.metrics`.
4. Printed the `imbalanced classification report` from `imblearn.metrics`.

**Conclusion:**

> Which model had the best balanced accuracy score?
>
> Which model had the best recall score?
>
> Which model had the best geometric mean score?

#### Ensemble Learning Technique

I trained and compared two different ensemble classifiers to predict loan risk and evaluate each model. I used the `balanced random forest classifier` and the `easy ensemble AdaBoost classifier`, and used 100 estimators for each. 

For each ensemble classifier:

1. Trained the model using the quarterly data from LendingClub.
2. Calculated the balanced accuracy score from `sklearn.metrics`.
3. Printed the confusion matrix from `sklearn.metrics`.
4. Generated a classification report using the `imbalanced_classification_report` from imbalanced learn.

**Results:**
<br/> 
Balanced Random Forest Model
<br/>
Parameters set for the model:
<br/>
![Doc File](Images/Blanced_Random_Forest-Parameters.png)
<br/>
![Doc File](Images/RF_matrix.png)
<br/>
<br/>
Imbalanced Calssification Report:
<br/>
![](Images/Balanced_Random_forest_CL-ClassificationReport.png)
<br/>
<br/>
Easy Ensemble Ada Boost
<br/>
Parameters set for the model:
<br/>
![Doc File](Images/EasyEnsemble-Parameters.png)
<br/>
![Doc File](Images/Easy_matrix.png)
<br/>
<br/>
Imbalanced Calssification Report:
<br/>
![Doc File](Images/Easy_ensemble-ClassificationReport.png)
<br/>
<br/>

**Conclusions:**
<br/>
![Doc File](Images/Both_compared.png)

> Which model had the best balanced accuracy score?
>
> Which model had the best recall score?
>
> Which model had the best geometric mean score?
>
> What are the top three features?
![Doc File](Images/Features_table.png)

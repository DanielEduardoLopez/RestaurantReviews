<p align="center">
	<img src="Images/Header.png?raw=true" width=80% height=80%>
</p>


# A Classification Model of Restaurant Reviews through Natural Language Processing
#### By Daniel Eduardo López

**[LinkedIn](https://www.linkedin.com/in/daniel-eduardo-lopez)**

**[Github](https://github.com/DanielEduardoLopez)**


____
### **1. Introduction**
**Text data** consists of phrases and sentences composed of words (Müller & Guido, 2016) that comes from a Natural Language, i.e., English, Spanish, Latin, etc. In this sense, **Natural Language Processing (NLP)** is the area of the computer science and artificial intelligence that deals with the processing and analysis of text data (Rogel-Salazar, 2020). 

The **bag-of-words model** is simple but effective representation of text data in which each word appearing in each text is counted and used to build a sparse matrix suitable to be used with Machine Learning (ML) techniques (Müller & Guido, 2016).

Some of the most common classification algorithms in ML are **Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Naive Bayes, Decision Trees, Random Forests, and XGBoost** (Müller & Guido, 2016; Ponteves, & Ermenko, 2021). 

According to Müller & Guido (2016), **random forests** are among the most popular ML techniques as they have a very good predictive power while reducing the overfitting. However, they are said to perform poorly on sparse datasets; being the linear models a more appropriate option (Müller & Guido, 2016).

In this context, it is desired to select the ML algorithm that is capable to yield the most accurate predictions on the NLP of restaurant reviews based on a bag-of-words model. 

____
### **2. General Objective**
To select the best machine learning algorithm for accurately classifying restaurant reviews into positive or negative through Natural Language Processing based on a bag-of-words model. 
____
### **3. Research Question**
Which machine learning algorithm for classifying restaurant reviews into positive or negative through Natural Language Processing based on a bag-of-words model is able to yield the highest accuracy?
____
### **4. Hypothesis**
**Random Forests** is the machine learning algorithm that yields the highest accuracy for classifying restaurant reviews into positive or negative through Natural Language Processing based on a bag-of-words model.
____
### **5. Abridged Methodology**
The methodology of the present study is based on Rollin’s Foundational Methodology for Data Science (Rollins, 2015):

1. **Analytical approach**: Building and evaluation of classification models.
2. **Data requirements**: Reviews of a restaurant and their corresponding labels (0 for negative and 1 for positive).
3. **Data collection**: Data was retrieved from <a href="https://www.kaggle.com/datasets/vigneshwarsofficial/reviews">Kaggle</a>.
4. **Data exploration**: Data was explored with Python 3 and its libraries Numpy, Pandas, Matplotlib and Seaborn.
5. **Data preparation**: Data was cleaned with Python 3 and its libraries Numpy, Pandas, Regular Expressions, and the Natural Language Toolkit.
5. **Data modeling**: First, a bag-of-words model was created from the text data. Then, the dataset was split in training, validation and testing sets. After that, Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Naive Bayes, Decision Trees, Random Forests, and XGBoost algorithms were used to build the models for classificating the restaurant reviews into positive or negative. The hyperparameters for each model were tunned using GridSearchCV or RandomizedSearchCV. Python 3 and its libraries Numpy, Pandas, and Sklearn were utilized for all the modeling steps.
6. **Evaluation**: The algorithms predictions were primarily evaluated through the accuracy rate, the area under the ROC curve (AUC ROC), and the root-mean-square error (RMSE). However, other metrics and tools such as confusion matrices, classification reports, AUC ROC plots, precision, negative predictive value (NPV), sensitivity, specificity, and the F1 score were also used.

___
### **6. Main Results**

#### **6.1 Data Collection**
As mentioned before, data about restaurant reviews and its corresponding labels was retrieved from <a href="https://www.kaggle.com/datasets/vigneshwarsofficial/reviews">Kaggle</a>.

#### **6.2 Data Exploration**
The data was explored to identify its general features and characteristics. In particular, dataset consisted of 1000 annotated reviews.

The amount of positive reviews was 500, and the amount of negative reviews was 500 too.

<p align="center">
	<img src="Images/Fig1_AmountReviews.png?raw=true" width=60% height=60%>
</p>

On the other hand, negative reviews tended to be longer than positive reviews.

<p align="center">
	<img src="Images/Fig2_TextLength.png?raw=true" width=60% height=60%>
</p>

Likewise, negative reviews tended to have more words than positive reviews.

<p align="center">
	<img src="Images/Fig3_NumberWords.png?raw=true" width=60% height=60%>
</p>

This suggests that disappointed customers tend to provide more details than happy customers.

<p align="center">
	<img src="Images/Fig4_NumberWordsBoxPlot.png?raw=true" width=60% height=60%>
</p>

Indeed, according to the boxplot, the negative reviews have more words than the positive ones.

Moreover, 2967 unique words were identified in the dataset.

The top 10 most frequent reviews were:

<p align="center">
	<img src="Images/Fig5_TopFrequentReviews.png?raw=true" width=60% height=60%>
</p>

On the other hand, the top 20 most frequent words were:

<p align="center">
	<img src="Images/Fig6_TopFrequentWords.png?raw=true" width=60% height=60%>
</p>

The latter insigth can also be conveyed through a Word Cloud:

<p align="center">
	<img src="Images/Fig7_WordCloud.png?raw=true" width=60% height=60%>
</p>


#### **6.3 Data Preparation**
The text was cleaned and  prepared for the subsequent modeling. To do so, abbreviations were converted to text. Then, only text characters were kept by using Regular Expressions. After that, text was transformed into lower case and split into words using lists. The words were later stemmed and the stop words were removed. Finally, the words were rejoined into a text string.

#### **6.4 Data Modeling**
A bag-of-words model was created in order to train several binary classification algorithms for classificating the restaurant reviews into positive or negative. The hyperparameters for each model were tunned using GridSearchCV or RandomizedSearchCV. The validation set was used to estimate the preliminary evaluation metrics for each model.

```python
# Logistic Regression Model
%%time

logreg_classifier = LogisticRegression(random_state = 0)

logreg_param_grid = {'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [1, 10, 100, 1000],
                    'tol': [1e-4, 1e-5, 1e-6],
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']                     
                    }

logreg_search = GridSearchCV(estimator = logreg_classifier,
                               param_grid = logreg_param_grid,
                               scoring = 'accuracy', # 'roc_auc'
                               cv = 5,
                               n_jobs = -1,                                                         
                               refit = True, 
                               verbose = True,
                               )

logreg_search.fit(X_train, y_train)
```

```bash
Model: LogisticRegression(random_state=0)


The best parameters are: {'C': 10, 'penalty': 'l1', 'solver': 'saga', 'tol': 0.0001}

              precision    recall  f1-score   support

           0       0.74      0.80      0.77        65
           1       0.78      0.71      0.74        63

    accuracy                           0.76       128
   macro avg       0.76      0.76      0.76       128
weighted avg       0.76      0.76      0.76       128

------------------------------

The best model yields an Accuracy of: 0.76316

The area under the ROC curve is: 0.75714

The RMSE is: 0.49213
```

```python
# K-Nearest Neighbors Model
%%time

KNN_classifier = KNeighborsClassifier()

KNN_param_grid = {'n_neighbors': list(range(3,50)),
                  'weights' : ['uniform','distance'],
                  'metric' : ['minkowski','euclidean','manhattan']                                   
                  }

KNN_search = GridSearchCV(estimator = KNN_classifier,
                          param_grid = KNN_param_grid,
                          scoring = 'accuracy', # 'roc_auc'
                          cv = 5,
                          n_jobs = -1,                                                         
                          refit = True, 
                          verbose = True,
                          )

KNN_search.fit(X_train, y_train)
```

```bash
Model: KNeighborsClassifier()


The best parameters are: {'metric': 'manhattan', 'n_neighbors': 12, 'weights': 'distance'}

              precision    recall  f1-score   support

           0       0.64      0.69      0.67        65
           1       0.66      0.60      0.63        63

    accuracy                           0.65       128
   macro avg       0.65      0.65      0.65       128
weighted avg       0.65      0.65      0.65       128

------------------------------

The best model yields an Accuracy of: 0.70914

The area under the ROC curve is: 0.64774

The RMSE is: 0.59293
```

```python
# Support Vector Machine Model
%%time

SVC_classifier = SVC(random_state = 0)

SVC_param_grid = {'C': [0.1,1, 10, 100], 
                  'gamma': [1,0.1,0.01,0.001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}

# RandomizedSearchCV was used beacause SVC is very computationally expensive
SVC_search = RandomizedSearchCV(estimator = SVC_classifier,
                                param_distributions = SVC_param_grid,
                                scoring = 'accuracy', # 'roc_auc'
                                cv = 5, 
                                n_jobs = -1,                                                                                
                                refit = True, 
                                verbose = True,
                                random_state = 0, 
                                n_iter = 50, # Number of samples
                              )

SVC_search.fit(X_train, y_train)
```

```bash
Model: SVC(random_state=0)


The best parameters are: {'kernel': 'rbf', 'gamma': 0.1, 'C': 10}

              precision    recall  f1-score   support

           0       0.77      0.78      0.78        65
           1       0.77      0.76      0.77        63

    accuracy                           0.77       128
   macro avg       0.77      0.77      0.77       128
weighted avg       0.77      0.77      0.77       128

------------------------------

The best model yields an Accuracy of: 0.77424

The area under the ROC curve is: 0.77326

The RMSE is: 0.47599
```

```python
# Naive Bayes Model
%%time

Bayes_classifier = GaussianNB()

Bayes_param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

Bayes_search = GridSearchCV(estimator = Bayes_classifier,
                          param_grid = Bayes_param_grid,
                          scoring = 'accuracy', # 'roc_auc'
                          cv = 5,
                          n_jobs = -1,                                                         
                          refit = True, 
                          verbose = True,
                          )

Bayes_search.fit(X_train, y_train)
```

```bash
Model: GaussianNB()


The best parameters are: {'var_smoothing': 0.1}

              precision    recall  f1-score   support

           0       0.89      0.52      0.66        65
           1       0.66      0.94      0.77        63

    accuracy                           0.73       128
   macro avg       0.78      0.73      0.72       128
weighted avg       0.78      0.73      0.71       128

------------------------------

The best model yields an Accuracy of: 0.72992

The area under the ROC curve is: 0.72979

The RMSE is: 0.52291
```

```python
# Decision Tree Model
%%time

tree_classifier = DecisionTreeClassifier(random_state = 0)

tree_param_grid = {
                  'criterion': ['gini', 'entropy', 'log_loss'],
                  'max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150, None],
                   'max_features': ['sqrt', 'log2', None]
                   }

tree_search = GridSearchCV(estimator = tree_classifier,
                          param_grid = tree_param_grid,
                          scoring = 'accuracy', # 'roc_auc'
                          cv = 5,
                          n_jobs = -1,                                                         
                          refit = True, 
                          verbose = True,
                          )

tree_search.fit(X_train, y_train)
```

```bash
Model: DecisionTreeClassifier(random_state=0)


The best parameters are: {'criterion': 'gini', 'max_depth': 15, 'max_features': None}

              precision    recall  f1-score   support

           0       0.68      0.91      0.78        65
           1       0.85      0.56      0.67        63

    accuracy                           0.73       128
   macro avg       0.77      0.73      0.72       128
weighted avg       0.76      0.73      0.73       128

------------------------------

The best model yields an Accuracy of: 0.73962

The area under the ROC curve is: 0.73162

The RMSE is: 0.51539
```

```python
# Random Forest Model
%%time

forest_classifier = RandomForestClassifier(random_state = 0)

forest_param_grid = {
                    'n_estimators': [100, 300, 500, 1000],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth' : [1, 5, 10, 20],
                    'max_features': ['sqrt', 'log2', None]
                    }

forest_search = GridSearchCV(estimator = forest_classifier,
                            param_grid = forest_param_grid,
                            scoring = 'accuracy', # 'roc_auc'
                            cv = 3, # Only 3 folds because RF are computationally expensive
                            n_jobs = -1,                                                         
                            refit = True, 
                            verbose = True,
                            )

forest_search.fit(X_train, y_train)
```

```bash
Model: RandomForestClassifier(random_state=0)


The best parameters are: {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'n_estimators': 1000}

              precision    recall  f1-score   support

           0       0.71      0.91      0.80        65
           1       0.87      0.62      0.72        63

    accuracy                           0.77       128
   macro avg       0.79      0.76      0.76       128
weighted avg       0.79      0.77      0.76       128

------------------------------

The best model yields an Accuracy of: 0.77286

The area under the ROC curve is: 0.76337

The RMSE is: 0.48412
```

```python
# XGBoost Model
%%time

xgb_classifier = XGBClassifier(objective= 'binary:logistic', random_state = 0)

xgb_param_grid = {
                    'n_estimators':[100, 300, 500],
                    'max_depth' : [1, 5, 10],                    
                    'learning_rate': [0.1, 0.01, 0.001]
                    }

# RandomizedSearchCV was used because XGB is somewhat computationally expensive
xgb_search = RandomizedSearchCV(estimator = xgb_classifier,
                                param_distributions = xgb_param_grid,
                                scoring = 'accuracy', # 'roc_auc'
                                cv = 3,
                                n_jobs = -1,                                                         
                                refit = True, 
                                verbose = True,
                                random_state = 0,
                                n_iter = 40
                                )

xgb_search.fit(X_train, y_train)
```

```bash
Model: XGBClassifier()


The best parameters are: {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.01}

              precision    recall  f1-score   support

           0       0.67      0.89      0.76        65
           1       0.83      0.54      0.65        63

    accuracy                           0.72       128
   macro avg       0.75      0.72      0.71       128
weighted avg       0.75      0.72      0.71       128

------------------------------

The best model yields an Accuracy of: 0.72992

The area under the ROC curve is: 0.71600

The RMSE is: 0.53033
```

#### **6.5 Evaluation**
The diferent fitted models were evaluated by using the testing set and primarily the following metrics: 
* Accuracy, 
* AUC ROC, and
* RMSE. 

Moreover, confusion matrices, classification reports, AUC ROC plots, precision, negative predictive value (NPV), sensitivity, specificity, and the F1 score were also used to assess the performance of each model.

In this sense, the accuracy, AUC, RMSE, precision, NPV, sensitivity, specificity and F1-score for each model are shown below:

Model | Accuracy | AUC | RMSE | Precision | NPV | Sensitivity | Specificity | F1
:--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--:
LogisticRegression | 0.71333 | 0.71572 | 0.53541 | 0.75714 | 0.675 | 0.67089 | 0.76056 | 0.71141
KNeighborsClassifier | 0.66667 | 0.67142 | 0.57735 | 0.73016 | 0.62069 | 0.58228 | 0.76056 | 0.64789
SVC | 0.73333 | 0.73614 | 0.5164 | 0.78261 | 0.69136 | 0.68354 | 0.78873 | 0.72973
GaussianNB | 0.67333 | 0.66349 | 0.57155 | 0.64423 | 0.73913 | 0.8481 | 0.47887 | 0.73224
DecisionTreeClassifier | 0.68667 | 0.70039 | 0.55976 | 0.92105 | 0.60714 | 0.44304 | 0.95775 | 0.59829
RandomForestClassifier | 0.74667 | 0.7545 | 0.50332 | 0.87273 | 0.67368 | 0.60759 | 0.90141 | 0.71642
XGBClassifier | 0.69333 | 0.7053 | 0.55377 | 0.88372 | 0.61682 | 0.48101 | 0.92958 | 0.62295

Furthermore, the confusion matrices for each model are shown below:

<p align="center">
	<img src="Images/Fig8_ConfusionMatrices.png?raw=true" width=85% height=85%>
</p>

From the confusion matrices above, it seems that **SVC** and **Random Forests** are the algorithms with the best performance, as they have the largest numbers of True Positives and True Negatives, as well as the lowest numbers of False Positives and False Negatives.

Moreover, in order to communicate the performance of each model for each evaluation metric, a heatmap was built:

<p align="center">
	<img src="Images/Fig9_ModelsPerformance.png?raw=true" width=60% height=60%>
</p>

From the heatmap above, **Random Forests, SVC and Logistic** are the algorithms with the best performance according to the **accuracy, AUC ROC and RMSE metrics**. They exhibited the highest accuracy and AUC ROC, as well as the lowest RMSE.

On the contrary, regarding the validity of the predictions, the best **precision** or **positive predictive value** corresponds to the **XGBoost and Decision Trees** algorithms. This means that their rate of accurate positive predictions is the highest or, in other words, they had the best ability to not to label as positive a review that is negative. Whereas the **Naive Bayes** algorithm yielded the highest rate of accurate negative predictions (highest **negative predictive value**) or, in other words, it had the best ability to not to label as negative a review that is positive.

On the other hand, regarding the completeness of the predictions, the **Naive Bayes** algorithm also exhibited the highest **sensitivity**, which means that this algorithm has the best ability to correctly classify true positive reviews from all the positive reviews or, in other words, it had the best ability to find all the positive reviews. Whereas the **XGBoost and Decision Trees** algorithms had the best ability to classify true negative reviews from all the negative reviews or, in other words, they had the best ability to find all the negative reviews (best **specificity**).

Finally, according to the **F1-score**, which is the harmonic mean of precision and sensitivity, the best model is the **Naive Bayes** algorithm.

After that, the accuracy, AUC ROC and RMSE was compared among all the models:

<p align="center">
	<img src="Images/Fig10_Accuracy.png?raw=true" width=60% height=60%>
</p>

Thus, in view of the above chart, the algorithm that yielded the **highest accuracy** was **Random Forest**.

<p align="center">
	<img src="Images/Fig11_AUCROC.png?raw=true" width=60% height=60%>
</p>

Thus, in view of the above chart, the algorithm that yielded the **highest AUC ROC** was **Random Forest**.

<p align="center">
	<img src="Images/Fig12_RMSE.png?raw=true" width=60% height=60%>
</p>

Thus, in view of the above chart, the algorithm that yielded the **lowest RMSE** was **Random Forest**.

<p align="center">
	<img src="Images/Fig13_ROC.png?raw=true" width=60% height=60%>
</p>

Finally, according the ROC curves, the algorithm that yielded the best results was **Random Forest** as its curve is arguably the closest to the y-axis, which means that this algorithm is capable to yield the highest true positive rate. On the other hand, it seems that the K-Neighbors was the worst algorithm as it is the closest to the x-axis, which represents the false positive rate.

Please refer to the **[Report](https://github.com/DanielEduardoLopez/RestaurantReviews/blob/main/Report.pdf)** for the full results and discussion.

___
### **7. Conclusions**
According to the combination of parameters tested, the **best model** for **classifying the reviews of a restaurant into positive or negative** through Natural Language Processing based on a bag-of-words model was the **Random Forest Classifier**, with an accuracy, AUC ROC, and RMSE of 0.75, 0.76, and 0.50, respectively. 

It is notable that this finding was in a contrary direction from what it is stated in the literature. This may suggest that either the Random Forest algorithms have been improved in the last couple of years or that the parameters used in the other algorithms were not adequate for the present classification task.

On the other hand, the second and third best models were **SVC** and **Logistic Regression**, according with the accuracy, AUC ROC, and RMSE metrics. This raises an apparent contradiction as the SVC model with the best performance used the radial basis function, which suggests that the classification problem is not linearly separable. 

In this context, as future research perspectives, further hyperparameter tunning is suggested on the Random Forest Classifier, SVC, and Logistic Regression algorithms, in order to find out whether the classification problem is linearly separable or not, as well as to reach a greater accuracy and a lower error. 

___
### **8. Bibliography**
- **Müller, A. C. & Guido, S. (2016)**. *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media. 
- **Ponteves, H. & Ermenko, K. (2021).** *Machine Learning de la A a la Z*. https://joanby.github.io/bookdown-mlaz/
- **Rogel-Salazar, J. (2020)**. *Advanced Data Science and Analytics with Python*. Chapman & Hall/CRC.
- **Rollins, J. B. (2015)**. *Metodología Fundamental para la Ciencia de Datos. Somers: IBM Corporation.* https://www.ibm.com/downloads/cas/WKK9DX51

___
### **9. Description of Files in Repository**
File | Description 
--- | --- 
NLP_RestaurantReviews.ipynb | Notebook with the Python code of the entire project.
NLP_RestaurantReviews.html | HTML version of the notebook with the Python code of the entire project.
Requirements.txt | Python requirements file.
Report.pdf | PDF version of the notebook with the Python code of the entire project.
Restaurant_Reviews.tsv | TSV file with the dataset.

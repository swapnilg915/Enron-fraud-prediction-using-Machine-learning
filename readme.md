Enron Fraud detection using Machine Learning algorithms

The aim of this project is to identify Enron employees who may have committed fraud based on the public Enron financial and email dataset. The provided (financial) dataset contains 21 features for each person. The email dataset contains the email text and author of the emails.There are total 121 data points in dataset amongst which 16 are POI's ( 13.2 %). By using machine learning on this dataset, we can find particular patterns in the data to detect the “person of Interest “, and hence we can detect the frauds involved.

In total I tried 6 algorithms. These are: 
1. Naive bayes 
2. DecisionTree 
3. Support vector Machine
4. K Nearest Nieghbors
5. Random Forest 
6. Adaboost. 

SVM and Random forest didn't worked well and they took more time for training compared to other algorithms.  Amongst all of these, KNN worked well compared to the others. KNN resulted in the precision of 0.57, recall of 0.39 and f1-score equal to 0.46. The classifier algorithm also gave the good accuracy of 0.86.

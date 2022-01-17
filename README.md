# UCU_AI_Course_Project

**The aim of our project** is to identify if a person has diabetes or is likely to have one based on the chosen features.

PDF presentation -- ./AI_Course_Project_Presentation.pdf

Video presentation -- [link](https://www.youtube.com/watch?v=BCX9xmVEBfg)


## Idea description

Today in Ukraine there are **1 million 134 thousand people** with diabetes, 181 thousand of which are forced to take insulin regularly.
Diabetes has been a problem for a long time, but why it is especially interesting for us now is because of COVID-19 and other diseases that,
when combined with diabetes, may lead to a sorrowful end. That is why it is mandatory to be able to identify your chronic diseases
as soon as possible so as to protect yourself from external stimuli.

In fact, sometimes people might not even know about their chronic diseases and might get acquainted with their diagnoses only when
lying in a hospital in the critical state. It is a pity, but such situations are not the rare cases. It would be very handy to get information 
about the likelihood of coming down with some chronic disease when you are doing the medical screening or, for example, undergoing the blood test,
which our model could help to implement. Thus, we could prevent a lot of people from facing such scenarios.

There is a hypothesis that it is possible to identify the likelihood that one has diabetes by checking the level of glucose and insulin.
We want to take it one step further and try to include more parameters to enhance the precision of this disease identification. 
**That is why the aim of our project is** to identify if a person has diabetes or is likely to have one based on the chosen features.


## Results

In this project we used ML models(RandomForestClassifier, DecisionTreeClassifier, SVM and XGBoost) and DL models(MLPClassifier and own NN), 
which tested on Pima Indians and Ranchi-835215. Our choice of models is based on the facts that we have small number of features in datasets, 
and also we chosen the best approaches from papers with similar datasets. Choice of datasets is based on logic and knowledge, 
which we learned during investigation of problem area. So here is the main results(look at ./best_results.csv,
deeper conclusions are in `03_Diabet_Prediction_Models.ipynb`):

* Based on F1 score for **Pima Indians dataset with random over-sampling**, **XGBClassifier** shown the best performance -- **0.851852** .
It can be explained in facts that dataframe has a small number of features, and idea of gradient boosting very good corresponds to our problem,
when we try to understand complicated disease pattern. Also we got good performance pruning after balancing train set.
<br/><br/>
* Based on F1 score for **Ranchi dataset with randon over-sampling**, **RandomForestClassifier** shown the best performance -- **0.948286**.
It can be explained in similar manner, but also in this dataset we have in several times higher correlation among features and target, hence,
Random Forest could understand this pattern better.

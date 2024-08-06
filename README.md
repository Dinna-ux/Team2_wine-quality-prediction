
# Wine Quality Prediction

This project has been developed as a submission for the Data Analytics Boot Camp, a collaborative educational initiative by Monash University and EdX.

Team Members:
Charles Morgan
Ben Mason
Witness Dinna
James Radford

# Project Overview

**Objective**

The aim of this project is to identify a Machine Learning Algorithm with the highest accuracy to evaluate our dataset and predict “good quality” wines. From this, our mock business case can be further developed to identify what qualities and characteristics make a good wine, reducing research time and production costs.

**Prerequisites**

Python : Ensure you have Python 3.8 or later installed.
Dependencies: Install the required Python packages 

**Running the Code**

Generate Visualizations
Run the following script to generate visualization `Exploratory_data_analysis_correlation.ipynb`
Train and Evaluate the Models: 
Run the following script to train and evaluate the model: `Machine_Learning_models.ipynb





**Code  Overview**
Data processing : The `CSV_to_SQL_Pipeline.ipynb `script includes functions for loading data, preprocessing, scaling features, training models, and evaluating their performance.
Visualisations: The `Exploratory_data_analysis_correlation.ipynb` script contains functions for creating plots and visualisations to better understand the data and model results.
Machine Learning Model Training and Evaluation: The script ‘Machine_Learning_Models’ evaluates 15 different Machine learning models and the script  ‘RandomForrestClassifier_Optimised’ is used to optimise the chosen model.


 # Data Engineering

Original data comes in the form of csv format. This was then read into Pandas and then into an SQLite database for use with in-memory data and ease of querying. Data was kept separate for Red and White wines for training and evaluation of the Machine Learning Algorithms.
Red Wine Quality Data: `red_wine_quality.db`
White Wine Quality Data: `white_wine_quality.db`

# Visualisations

Heatmap:
Correlation data show that the strongest positive relationship between red wine quality is alcohol content, followed by sulfates. The strongest negatively correlated component was volatile acidity. 
Similarly in white wine, alcohol content has the strongest positive correlation followed by PH index. The strongest negatively correlated component was density.

![alt text](https://github.com/Dinna-ux/Team2_wine-quality-prediction/blob/main/Images/Heatmap%20red%20wine.png)
                                      Figure 1: Red Wine Heatmap


![alt text](https://github.com/Dinna-ux/Team2_wine-quality-prediction/blob/main/Images/Heatmap%20White%20wine.png)
                                      Figure 2: White Wine Heatmap















Boxplots:
Box Plots were used to find the range of each component of the dataset. Red wine shows the greatest variance from alcohol content followed by sulphates shown in Figure 4. White has the greatest variance of citric acid followed closely by alcohol content shown in Figure 5.

To avoid overfitting, prior to the model training, each feature was scaled due to the fact that a large value in one variable can dominate over other variables during the training process. 

![alt text](https://github.com/Dinna-ux/Team2_wine-quality-prediction/blob/main/Images/Box%20Plot%20Red%20wine.png)
Figure 4: Boxplots for each component range Red wine






![alt text](https://github.com/Dinna-ux/Team2_wine-quality-prediction/blob/main/Images/Box%20plot%20white%20wine.png)
Figure 5: Boxplots for each component range White wine










# Model Training and Evaluation 
The script ‘Machine_Learning_Models’ evaluates 15 different Machine learning models shown below, through which Random Forest Classifier was selected as the most accurate model with precision at 0.85, recall at 0.64, F1 at 0.73 and accuracy at 0.89.

Random Forest Classifier was an obvious choice as it is known to be one of the stronger models for categorical data. A high precision value was considered favorable as a False Positive prediction would be detrimental to our business. The case where we have manufactured and priced a poor quality wine as a high quality wine was considered to be the most determining factor as this would lead to poor brand image and low customer retention.

 From here, the script ‘RandomForrestClassifier_Optimised’  trains the Random Forest Classifier and performs hyperparameter tuning using GridSearchCV. It also evaluates and compares the performance of the initial and optimized models.
 

![alt text](https://github.com/Dinna-ux/Team2_wine-quality-prediction/blob/main/Images/models%20tested.PNG)
Table 1: ML models Tested










# Results:
The project includes various metrics for model performance, including accuracy, precision, recall, F1, score, and confusion matrices. The results for both initial and optimized models are summarized in an output of `wine_quality_results_RFC.csv`

The optimisation process did not prove fruitful with only a slight improvement on the Red wine dataset as shown in the table below. White wine dataset actually declined in performance. This could be due to overfitting, sub optimal hyperparameters or inherent differences within the dataset. The next step would be to perform some more advanced optimisation analysis such as Bayesian Optimisation. 

![alt text](https://github.com/Dinna-ux/Team2_wine-quality-prediction/blob/main/Images/results%20table.PNG)
Table 2: Initial and Optimised Results from Random Forest Classifier

# Conclusion

This project successfully demonstrated the application of machine learning algorithms to predict wine quality based on chemical properties. The Random Forest Classifier, after hyperparameter tuning, provided the best performance for both red and white wines. These insights can be valuable for wine producers to enhance wine quality and optimize production processes.

**Acknowledgements**

We would like to thank Monash University and EdX for providing the platform and resources for this project. Special thanks to our instructors, TAs and mentors for their guidance and support throughout the Data Analytics Boot Camp.


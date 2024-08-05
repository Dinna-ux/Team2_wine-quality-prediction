# Wine Quality Prediction

This project has been developed as a submission for the Data Analytics Boot Camp, a collaborative educational initiative by Monash University and EdX.

Team Members:
Charles Morgan
Ben Mason
Witness Dinna
James Radford

Project Overview

Objective

The aim of this project is to identify a Machine Learning Algorithm with the highest accuracy to evaluate our dataset and predict “good quality” wines. From this, our mock business case can be further developed to identify what qualities and characteristics make a good wine, reducing research time and production costs.

Data Processing 
Original data comes in the form of csv format. This was then read into Pandas and then into an SQLite database for use with in-memory data and ease of querying. Data was kept separate for Red and White wines for training and evaluation of the Machine Learning Algorithms.
Visualisations
Heatmap:
Correlation data show that the strongest positive relationship between red wine quality is alcohol content, followed by sulphates. Similarly in white wine, alcohol content has the strongest correlation followed by PH index. 

 
Figure 1: Red Wine Heatmap


 
Figure 2: White Wine Heatmap

<h1 class="code-line" data-line-start=0 data-line-end=1><a id="MachineLearning_Models_on_Different_Scenarios_0"></a>Machine-Learning Models on Different Scenarios</h1>

<p class="has-line-data" data-line-start="2" data-line-end="3">These Notebooks with their Question Statement and Reports, came under the course CSL2050, taken by Prof. Richa Singh.</p>
<h2 class="code-line" data-line-start=4 data-line-end=5><a id="Lab1__Confusion_Matrix_4"></a>Lab-1 :- Confusion Matrix</h2>

<p class="has-line-data" data-line-start="6" data-line-end="7"><img src="https://user-images.githubusercontent.com/54277039/139558169-45b8fcaf-3f17-4828-8619-f323d319bdd6.png" alt="image" height="400" width="1000" align="center"></p><p>A csv file has been provided to you. It contains three columns. First column is the actual labels for a binary classification problem. Second, and third column are predicted probabilities from two classifiers. You will be converting these probabilities values in the final label based on the threshold value. Helping code-script is in Notebook. You are supposed to complete the functions computing the different evaluation metrics as described in the Colab Notebook at this link. You may download the Notebook and may start working on it on your local device or on Colab Notebook. The Notebook is provided to you for a quick start. You will define the functions for the following tasks</p>
<p>i.)  To calculate accuracy.</p>
<p>ii.) To calculate precision and recall.</p>
<p>iii.) To calculate F1 score.</p>

<p>Both per-class and per-sample average precision, recall and F1-scores need to be calculated.</p>
<p>Additionally you are required to change the threshold value (0.5, 0.4, 0.6 etc.) and compare, contrast the difference in metrics for both the models.</p>
<h2 class="code-line" data-line-start=8 data-line-end=9><a id="Lab2__Decision_Tree_8"></a>Lab-2 :- Decision Tree</h2>

<p class="has-line-data" data-line-start="10" data-line-end="11"><img src="https://user-images.githubusercontent.com/54277039/139558187-d228ae82-86be-4f06-9837-dbe6381cd45d.png" alt="image" height="400" width="1000" align="center">
<p>Q1: A csv file has been provided to you. The dataset represents the mood of a student to go to class depending on the weather at IIT Jodhpur. We have been accustomed to online classes so this is to give you a feeling of attending classes in the post-COVID scenario. A Colab Notebook is attached for your reference about the stepwise procedure to solve the exercise . The tasks are as follows:</p>
<p>i)  Preprocessing the data.</p>
<p>ii) Cross-validation over the data.</p>
<p>iii) Training the final model after cross-validation</p>
<p>iv) Perform decision tree classification and calculate the prediction accuracy for the test data.</p>
<p>v) Plot the decision tree and the decision surface.</p>

<p>Q2: In the previous case, the nodes are split based on entropy/gini impurity.The following dataset contains real-valued data..The column to be predicted is 'Upper 95% Confidence Interval for Trend' i.e. the last column present in the dataset using other columns as features. The tasks are as follows:</p> 
<p>i)  Preprocessing the data.</p>
<p>ii) Cross-validation over the data.</p>
<p>iii) Training the final model after cross-validation</p>
<p>iv) Perform decision tree regression and calculate the squared error between the predicted and the ground-truth values for the test data.</p>
<p>v) Plot the decision tree and the decision surface.</p>
<h2 class="code-line" data-line-start=12 data-line-end=13><a id="Lab3__Random_Forest_and_Bagging_Classifier_12"></a>Lab-3 :- Random Forest and Bagging Classifier</h2>

<p class="has-line-data" data-line-start="14" data-line-end="16"><img src="https://user-images.githubusercontent.com/54277039/139558195-6f6ce0af-ac1a-4f52-8895-57d55daa20c6.png" alt="image" height="400" width="1000" align="center"><br></p>
<p>Consider the credit sample dataset, and predict whether a  customer will repay their credit within 90 days. This is a binary classification problem; we will assign customers into good or bad categories based on our prediction.</p>
<p>Data Description:-</p>
<p>Features --> Variable Type	--> Value Type --> Description</p>
<p></p>
<p>Age --> Input Feature --> integer --> Customer age</p>
<p>Debt Ratio --> Input Feature --> real --> Total monthly loan payments (loan, alimony, etc.) / Total monthly income percentage.</p>
<p>Number_Of_Time_30-59_Days_Past_Due --> Input Feature --> integer --> The number of cases when a client has overdue 30-59 days (not worse) on other loans during the last 2 years.</p>
<p>Number_Of_Time_60-89_Days_Past_Due --> Input Feature --> integer --> A number of cases when the customer has 60-89dpd (not worse) during the last 2 years.</p>
<p>Number_Of_Times_90_Days_Late --> Input Feature --> integer --> Number of cases when a customer had 90+dpd overdue on other credits</p>
<p>Dependents --> Input Feature --> integer --> The number of customer dependents</p>
<p>Serious_Dlq_in_2yrs --> Target Variable	 --> Binary: 0 or 1 --> The customer hasn't paid the loan debt within 90 days</p>
<p>Perform the following tasks for this dataset:-</p>
<p>Question-1 (Random Forest):</p>
<p>1. Preprocessing the data.</p>
<p>2. Plot the distribution of the target variable.</p>
<p>3. Handle the NaN values.</p>
<p>4. Visualize the distribution of data for every feature.</p>
<p>5. Train the Random Forest Classifier with the different parameters, for e.g.:-</p>
<p>     Max_features = [1,2,4]</p>
<p>     Max_depth = [2,3,4,5]</p>
<p>6. Perform 5 fold cross-validation and look at the ROC AUC against different values of the parameters (you may use Stratified KFold function for this) and Perform the grid-search for the parameters to find the optimal value of the parameters. (you may use GridSearchCV for this )</p>
<p>7. Get the best score from the grid search.</p>
<p>8. Find the feature which has the weakest impact in the Random Forest Model.</p>
<p><img src="https://user-images.githubusercontent.com/54277039/139558203-ecdb25a1-ecea-47f8-ab04-f3c49b323550.png" alt="image" height="400" width="1000" align="center"></p>
<p>Question-2 (Bagging) :</p>
<p>1. Perform bagging-based classification using Decision Tree as the base classifier.</p>
<p>2. The number of trees to be considered is {2,3,4}.</p> 
<p>3. Perform 5 fold cross-validation using ROC AUC metric to evaluate the models and collect the cross-validation scores (use function cross_val_score for this).</p>
<p>4. Summarize the performance by getting mean and standard deviation of scores</p>
<p>5. Plot the model performance for comparison using boxplot.</p>
<p>6. Compare the best performance of bagging with random forest by plotting using boxplot.</p>

<h2 class="code-line" data-line-start=17 data-line-end=18><a id="Lab4__Adaboost_and_Bayes_Classification_17"></a>Lab-4 :- Adaboost and Bayes Classification</h2>

<p class="has-line-data" data-line-start="19" data-line-end="21"><img src="https://user-images.githubusercontent.com/54277039/139558212-21270692-195b-44f9-9985-921207523a1d.png" alt="image" height="400" width="1000" align="center"><br></p>
<p>Perform the following tasks for this dataset:-</p>
<p>Question-1 (Boosting ):</p>
<p>1. Preprocessing the data.</p>
<p>2. Plot the distribution of the target variable.</p>
<p>3. Visualize the distribution of data for every feature.</p>
<p>4. Perform boosting-based classification using Decision Tree as the base classifier.</p> 
<p>5. Perform cross validation over the data and calculate accuracy for a weak learner.</p>
<p>6. Build the AdaBoost model using the weak learner by increasing the number of trees from 1 to 5 with a step of 1. Compute the model performance.</p>
<p>Question-2 (Bayes classification) :</p>
<p>1. Estimate the accuracy of Naive Bayes algorithm using 5-fold cross validation on the data set. Plot the ROC AUC curve for different values of parameters.</p>
<p>2. Use linear discriminant function to calculate the  accuracy on the classification task with 80% training and 20% testing data.</p>
<p>3. Calculate the Bayes risk from the customized matrix of your choice.</p>
<p><img src="https://user-images.githubusercontent.com/54277039/139558271-94c0429a-9cb4-4def-9307-50b53b760dc0.png" alt="image" height="400" width="1000" align="center"></p>
<p>Question 3: Visualisation in Bayesian Decision Theory</p>
<p>DATASET 1:</p>
<p>1. Consider the height of the car and its cost is given. If the cost of a car > 550 then the label is 1, otherwise 0.</p>
<p>2. Create the labels from the given data.</p>
<p>3. Plot the distribution of samples using histogram.</p>
<p>4. Determine the prior probability for both the classes.</p>
<p>5. Determine the likelihood / class conditional probabilities for the classes. (Hint : Discretize the car heights into bins, you can use normalized histograms)</p>
<p>6. Plot the count of each unique element for each class. (Please mention in the report why this plot is different from the distribution)</p>
<p>7. Calculate the P(C1|x) and P(C2|x) i.e posterior probabilities and plot them in a single graph.</p>
 
<p>DATASET 2:</p>
<p>Now for the second dataset there are two files c1 and c2 . c1 and c2 contain two features each for class 1 and 2 respectively. Read the dataset and repeat all the above steps for Dataset 2.</p>
<p>Note : Plot the data distribution and the histogram of feature 1 and feature 2 in the X axis and Y axis respectively. The distribution of feature 1 will be along the top of X axis and feature 2 along the right of Y axis. An example is shown below.</p>

<p>Real Life Dataset: </p>
<p>Now it's time to visualise a real life dataset. Take any one feature from the above IRIS dataset and take the class labels. In this dataset there are three class labels. Extend all the visualisation mentioned previously for this dataset.</p>

<h2 class="code-line" data-line-start=22 data-line-end=23><a id="Lab5__Text_Analysis_using_Bayes_Classification_22"></a>Lab-5 :- Text Analysis using Bayes Classification</h2>

<p class="has-line-data" data-line-start="24" data-line-end="25"><img src="https://user-images.githubusercontent.com/54277039/139558280-08886ebc-e8ec-40ff-a6bb-86f0b84be3d3.png" alt="image" height="400" width="1000" align="center"></p>
<p>Data Preparation:</p>
<p>1. Import necessary libraries</p>
<p>2. Load the data</p>
<p>3. Plot the count for each target</p>
<p>4. Print the unique keywords</p>
<p>5. Plot the count of each keyword</p>
<p>6. Visualize the correlation of the length of a tweet with its target</p>
<p>7. Print the null values in a column</p>
<p>8. Removing null values</p>
<p>9. Removing Double Spaces, Hyphens and arrows, Emojis, URL, another   Non-English or special symbol</p>
<p>10. Replace wrong spellings with correct ones</p>
<p>11. Plot a word cloud of the real and fake target</p>
<p>12. Remove all columns except text and target</p>
<p>13. Split data into train and validation</p>
<p>14. Compute the Term Document matrix for the whole train dataset as well as for the two classes. </p>                                                                         
<p>15. Find the frequency of words in class 0 and 1. </p>
<p>16. Does the sum of the unique words in target 0 and 1 sum to the total number of unique words in the whole document? Why or why not?</p>
<p>17. Calculate the probability for each word in a given class.</p>
<p>18. We have calculated the probability of occurrence of the word in a class, we can now substitute the values in the Bayes equation. If a word from the new sentence does not occur in the class within the training set, the equation becomes zero. This problem can be solved using smoothing like Laplace smoothing. Use Bayes with Laplace smoothing to predict the probability for sentences in the validation set.</p>
<p>19. Print the confusion matrix with precision, recall and f1 score.</p>

<h2 class="code-line" data-line-start=26 data-line-end=27><a id="Lab6__Linear_Regression_26"></a>Lab-6 :- Linear Regression</h2>

<p class="has-line-data" data-line-start="28" data-line-end="29"><img src="https://user-images.githubusercontent.com/54277039/139558285-43b8bbe2-7ef1-4e62-afe8-47716abbe0bd.png" alt="image" height="400" width="1000" align="center"></p>
<p>Build a linear regression model for the Medical cost dataset. The dataset consists of age, sex, BMI(body mass index), children, smoker, and region features, and charges. You need to predict individual medical costs billed by health insurance. The target variable here is charges, and the remaining six variables such as age, sex, BMI, children, smoker, region, are the independent variables. The hypothesis function looks like</p>
<p>hθ(xi)=θ0+θ1age+θ2sex+θ3bmi+θ4children+θ5smoker+θ6region</p>
<p>Perform the following tasks for this dataset:-</p>
<p>1. Load the dataset and do exploratory data analysis.</p>
<p>2. Plot correlation between different variables and analyze whether there is a correlation between any pairs of variables or not.</p>
<p>4. Plot the distribution of the dependent variable and check for skewness (right or left skewed) in the distribution.</p>
<p>5. Convert this distribution into normal by applying natural log and plot it. (If the distribution is normal then skip this).</p>
<p>6. Convert categorical data into numbers. (You may choose one hot encoding or label encoding for that).</p>
<p>7. Split the data into training and testing sets with ratio 0.3.</p>
<p>8. Build a model using linear regression equation θ=(XTX)−1XTy . (First add a feature X0 =1 to the original dataset).</p>
<p>9. Build a linear regression model using the sklearn library. ( No need to add X0 =1, sklearn will take care of it.) </p>
<p>10. Get the parameters of the models you built in step 7 and 8, compare them, and print comparisons in a tabular form. If the parameters do not match, analyze the reason(s) for this (they should match in the ideal case).</p>
<p>11. Get predictions from both the models (step 7 and step 8).</p>
<p>12. Perform evaluation using the MSE of both models (step 7 and step 8). (Write down the MSE equation for the model in step 7 and use the inbuilt MSE for the model in step 8).</p>
<p>13. Plot the actual and the predicted values to check the relationship between the dependent and independent variables. (for both the models)</p>

<h2 class="code-line" data-line-start=30 data-line-end=31><a id="Lab7__MultiLayer_Perceptron_KMeans_Clustering_and_Neural_Network_30"></a>Lab-7 :- Multi-Layer Perceptron, K-Means Clustering and Neural Network</h2>

<p class="has-line-data" data-line-start="32" data-line-end="34"><img src="https://user-images.githubusercontent.com/54277039/139558298-1dc2aa04-b20b-43eb-81ce-43c15c5793f0.png" alt="image" height="400" width="1000" align="center"><br></p>
<p>The objective of this assignment is to learn to implement Multi Layer Perceptron (MLP) from scratch using python. For this a nice tutorial has been provided. After implementing MLP from scratch, you need to compare it with Sklearn’s in-built implementation (resource-2). For this you are supposed to use wheat seeds dataset provided.</p>
<p>Please go through the following blog to learn how to recognize handwritten digits using Neural Network. Here Neural Network is coded using PyTorch Library in Python.</p>
<p>Use above code and report your observation based on the following: </p>
<p>(i) Change loss function,</p>
<p>(ii) Change in learning rate, and</p>
<p>(iii) Change in Number of hidden layers</p>
<p><img src="https://user-images.githubusercontent.com/54277039/139558312-16a09aa0-7d69-44eb-9591-fab6c80399d9.png" alt="image" height="400" width="1000" align="center"></p>
<p>You may use the MNIST dataset or any dataset for Face Images or Flower Images or Iris dataset for this Question.</p>
<p>Implement k-means clustering. Analyse the clusters formed for various values of k. Display the centroids of the clusters. DO NOT USE IN_BUILT ROUTINE for k-means clustering.</p>
<h2 class="code-line" data-line-start=35 data-line-end=36><a id="Lab8__Dimensionality_Reduction_and_Feature_Selection_35"></a>Lab-8 :- Dimensionality Reduction and Feature Selection</h2>

<p class="has-line-data" data-line-start="37" data-line-end="39"><img src="https://user-images.githubusercontent.com/54277039/139558346-9803c9a8-d3dd-4d0a-a1b7-3b625c07c683.png" alt="image" height="400" width="1000" align="center"><br></p>
<p>Using the data set, execute a PCA analysis using at least two dimensions of data (note that the last column should not be used here). In your code , discuss/include the following items.</p>
<p>1. Standardize the data.</p>
<p>2. How many eigenvectors are required to preserve at least 90% of the data variation?</p>
<p>3. Look at the first eigenvector. What dimensions are the primary contributors to it (have the largest coefficients)? Are those dimensions negatively or positively correlated? </p>
<p>4. Show a plot of your transformed data using the first two eigenvectors. </p>
<p>For the aforementioned dataset perform Linear discriminant analysis </p>
<p>1. Compare the results of PCA and LDA. </p>
<p>2. Plot the distribution of samples using the first 2 principal components and the first 2 linear discriminants.</p>
<p>3. Learn a Bayes classifier using the original features and compare its performance with the features obtained in part (b).</p>

<p><img src="https://user-images.githubusercontent.com/54277039/139558360-2f2e68a4-8992-4c19-b2dc-cdd94e4a639d.png" alt="image" height="400" width="1000" align="center"></p>
<p>Perform feature selection using any 2  methods studied in class and do the  classification for the dataset using a classification algorithm of your choice. Do the following tasks: </p>
<p>1. Preprocess the data and perform exploratory data analysis.</p>
<p>2. Identify the  features having high significance using both of the methods.</p>
<p>3. Calculate and compare the  accuracy and F1 score by both the methods and with the classifier learned using all the features (without doing feature selection), and analyze which method performs the best and why.</p>
<p>4. Use Pearson Correlation and compute correlated features with a threshold of 70%.</p>
<h2 class="code-line" data-line-start=40 data-line-end=41><a id="Lab9__Support_Vector_Machines_40"></a>Lab-9 :- Support Vector Machines</h2>
<p class="has-line-data" data-line-start="42" data-line-end="43"><img src="https://user-images.githubusercontent.com/54277039/139558368-198043d9-44b3-447e-b039-af0a12541f08.png" alt="image" height="400" width="1000" align="center"></p>
<p>Problem 1 (Handwritten Digit Classification):</p>
<p>Your goal is to develop a Handwritten Digit Classification model. This model should take input as an image of a Handwritten Digit and classify it to one of the five classes {0,1,2,3,4}. To this end, you are supposed to work on the MNIST dataset. You were shown how the MNIST dataset is read and displayed in python in one of the labs. Now, perform the following experiments: </p>

<p>1. Use 70-20-10 split for training, validation and testing. Use the validation set for hyperparameter tuning by doing grid search, and report the classification accuracy on the test set. </p>
<p>2. Use nearest neighbour, perceptron and SVM classifiers for classifying handwritten digits of MNIST, and compare their performance. </p>
<p>3. Normalize the data by mean subtraction followed by standard deviation division. Redo the above experiments on normalized data and report the performance. </p>
<p>4. Implement any two from OVA/OVO/DAG, and compare the results. </p>

<p>Problem 2:  Use the “diabetes” dataset from the previous lab assignment. Split the dataset into training, validation and test sets (e.g., in 70:20:10 split, or 80:10:10 split). </p>
<p>On this dataset, evaluate the classification accuracy using the following classifiers:</p>
<p>1. SVM classifier (using a linear kernel)</p>
<p>2. SVM classifier (using a Polynomial kernel and a Gaussian kernel)</p>
<p>3. If your data is not linearly separable, then you may use the soft margin SVM formulation. You can use the inbuilt implementation of SVM in SciKit Learn. </p>
<p>4. Compare and analyze the results obtained in different cases. During cross-validation, try different values of various hyper-parameters, such as the regularization hyper-parameter ‘C’ (e.g., by varying it in {0.0001, 0.001, ... , 1, 10, 100, 1000}), and the kernel function hyper-parameter(s). </p>
<p>5. Report the number of support vectors obtained in the final model in each case. </p>
<p>6. Perform an experiment to visualize the separating hyper-plane (in a 2-D space). </p>


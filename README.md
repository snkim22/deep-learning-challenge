# deep-learning-challenge

### Instructions

**Step 1: Preprocess the Data**

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset.

**Step 2: Compile, Train, and Evaluate the Model**

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

**Step 3: Optimize the Model**

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

**Step 4: Write a Report on the Neural Network Model**

### Report:

#### Overview:
  - The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. Construct a deep learning neural network model that could accurately classify whether an organization would thrive under Alphabet Soup's support.
  - 
#### Results: 

#### Data Preprocessing
  - The `IS_SUCCESSFUL` variable was used as the target variable to serve as a binary indicator of an applicant's success after receiving funding.
  - Features set: `APPLICATION_TYPE`,`AFFILIATION—Affiliated`,`CLASSIFICATION`,`USE_CASE`,`ORGANIZATION`,`STATUS`,`INCOME`,`SPECIAL_CONSIDERATIONS`,`ASK_AMT`
  - The `EIN` and `NAME` cariables were removed as they were not pertinent to our predicitve task
    
#### Compiling, Training, and Evaluating the Model
  - The neural network consistes of two hidden layers - the first with 10 neurons and the second with 15 neurons.
  - The input layer is defined by the number of input features.
  - `ReLU` was used as the activation function for the hidden layers beucase it had mitigate for vanishing gradient.
  - This model did not meet the 75% accuracy mark -> 72.7%.
  - Three additional attempts were made to optimize the model:
    1.Reduce number of epochs to training regimen -> Accuracy: 72.41%
    2.Increase neurons to hidden layer -> Accuracy: 72.33%
    3.Using different activation function for hidden layers -> Accuracy: 72.37%

#### Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.`
  - The model achieved 72.7% accuracy, falling short of the 75% goal. Optimization attempts, such as modifying training epochs and activation functions, didn't significantly enhance performance.
  - Exploring advanced architectures sucha as `ensemble methods` or implementing `Keras Tuner` to find the optimal hyperparameters may be a fruitful next step to improve accuracy.

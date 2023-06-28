# Sleep-Deprived-Student-Team ☻

## ❀ Our Group Member : ❀
- Adinda Rahajeng S.P &nbsp;&nbsp;&nbsp;2141720158
- Aminy Ghaisan N &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2141720163
- Faradhisa Aldina P &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  2141720159
- Fina Salama Q.H &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  2141720164
- Zerlina Wollwage&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2141720146
---
### Background
Over time, the medicine’s effective time shortens, causing discomfort among PD patients. Thus, PD patients and clinicians must monitor and record the patient symptom changes for adequate treatment. Parkinson's disease (PD) is a slowly progressive nervous system condition caused by the loss of dopamine-producing brain cells. It primarily affects the patient's motor abilities, but it also has an impact on non-motor functions over time. Patients' symptoms include tremors, muscle stiffness, and difficulty walking and balancing. Then it disrupts the patients' sleep, speech, and mental functions, affecting their quality of life (QoL). With the improvement of wearable devices, patients can be continuously monitored with the help of cell phones and fitness trackers. We use these technologies to monitor patients' health, record WO periods, and document the impact of treatment on their symptoms in order to predict the wearing-off phenomenon in Parkinson's disease patients.

---

### Task
Using multiple linear regression, try to find the best regression model that is suitable to represent the case: **helping the doctors to create specific treatment strategies to manage Parkinson's disease and its associated symptoms properly**. It means you are asked to create a model that can anticipate the "wearing-off" of anti-Parkinson Disease medication.

---
### Results from our group:
* **★ Import Modules and Libraries ★**
  &nbsp;&nbsp;
  ```
  import pandas as pd
  from sklearn import linear_model
  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np
  import statsmodels.api as sm
  ```
  The code above is used to import some Python libraries for data analysis and machine learning. Libraries are:
  * **pandas**: library for data manipulation and analysis.
  * **sklearn**: library for machine learning and data mining.
  * **matplotlib.pyplot**: library for data visualization.
  * **seaborn**: library for data visualization based on matplotlib.
  * **numpy**: library for numeric computation.
  * **statsmodels.api**: library for statistical modeling.

    After importing this library, we can use its functions and methods to perform various tasks such as reading, cleaning, and     
  analyzing data, creating and evaluating machine learning models, and creating data and model visualizations. Overall, these libraries 
  provide a powerful toolkit for data analysis and machine learning in Python.

* **★ Import CSV Dataset ★**
  &nbsp;&nbsp;
  ```
  df = pd.read_csv('combined_data(1).csv')
  df
  ```
  Code **df = pd.read_csv('combined_data(1).csv')** reads a CSV file named **"combined_data(1).csv"** and saves it as a pandas     
  DataFrame called **"df"**. Once the data is loaded into the DataFrame, we can use pandas and other libraries to manipulate, analyze 
  and visualize the data.

* **★ Heart Rate VS Wearing Off Duration Scatter Plot ★**
  &nbsp;&nbsp;
  ```
  plt.scatter(df['heart_rate'], df['wo_duration'])
  plt.title('Heart Rate VS Wearing Off Duration')
  plt.xlabel('Heart Rate')
  plt.ylabel('Wearing Off Duration')
  plt.grid(True)
  plt.savefig("HeartRate.jpg")
  plt.show()
  ```
  The purpose of the code is to create a scatter plot of two variables, "heart_rate" and "wo_duration", from a pandas DataFrame called    "df". The scatter plot is then saved as an image file named "HeartRate.jpg". Here is a summary of what each line of the code does:
  * **plt.scatter(df['heart_rate'], df['wo_duration'])**
      This line creates a scatter plot with **"heart_rate"** on the x-axis and **"wo_duration"** on the y-axis. The data for these            variables is taken from the pandas DataFrame **"df".**
  * **plt.title('Heart Rate VS Wearing Off Duration')**
      This line sets the title of the plot to **"Heart Rate VS Wearing Off Duration"**
  * **plt.xlabel('Heart Rate')**
      This line sets the label for the x-axis to **"Heart Rate".**
  * **plt.ylabel('Wearing Off Duration')**
      This line sets the label for the y-axis to **"Wearing Off Duration".**
  * **plt.grid(True)**
      This line adds a grid to the plot.
  * **plt.savefig("HeartRate.jpg")**
      This line saves the plot as an image file named **"HeartRate.jpg".**
  * **plt.show()**
      This line displays the plot on the screen.
    
  Overall, This plot can be used to visualize the relationship between the two variables and to identify any patterns or trends in the     data.

* **★ Stress Score VS Wearing Off Duration Scatter Plot ★**
  &nbsp;&nbsp;
  ```
  plt.scatter(df['stress_score'], df['wo_duration'])
  plt.title('Stress Score VS Wearing Off Duration')
  plt.xlabel('Stress Score')
  plt.ylabel('Wearing Off Duration')
  plt.grid(True)
  plt.savefig("StressScore.jpg")
  plt.show()
  ```
   The code is used to create a scatter plot of two variables, **"stress_score"** and **"wo_duration"**, from a pandas DataFrame called **"df"**. The scatter plot is then saved as an image file name  **"StressScore.jpg"**. Here is a summary of what each line of the code does:
  * **plt.scatter(df['stress_score'], df['wo_duration'])**
      This line creates a scatter plot with **"stress_score"** on the x-axis and **"wo_duration"** on the y-axis.       The data for these variables is taken from the pandas DataFrame **"df".**
  * **plt.title('Stress Score VS Wearing Off Duration')**
      This line sets the title of the plot to **"Stress Score VS Wearing Off Duration".**
  * **plt.xlabel('Stress Score')**
      This line sets the label for the x-axis to **"Stress Score".**
  * **plt.ylabel('Wearing Off Duration')**
      This line sets the label for the y-axis to **"Wearing Off Duration".**
  * **plt.grid(True)**
      This line adds a grid to the plot.
  * **plt.savefig("StressScore.jpg")**
      This line saves the plot as an image file named **"StressScore.jpg".**
  * **plt.show()**
      This line displays the plot on the screen.

  Same as before, This plot can be used to visualize the relationship between the two variables and to identif  any patterns or trends in the data.

* **★ Time From Last Drug Taken VS Wear Off Duration Scatter Plot ★**
  &nbsp;&nbsp;
  ```
  plt.scatter(df['time_from_last_drug_taken'], df['wo_duration'])
  plt.title('Time From Last Drug Taken VS Wear Off Duration')
  plt.xlabel('Time From Last Drug Taken')
  plt.ylabel('Wear Off Duration')
  plt.grid(True)
  plt.savefig("DrugTime.jpg")
  plt.show()
  ```
  The code is used to create a scatter plot of two variables, **"time_from_last_drug_taken"** and **"wo_duration"**, from a pandas DataFrame called **"df"**. The scatter plot is then saved as an image file named **"DrugTime.jpg"**. Here is a summary of what each line of the code does:
  * **plt.scatter(df['time_from_last_drug_taken'], df['wo_duration'])**
      This line creates a scatter plot with **"time_from_last_drug_taken"** on the x-axis and **"wo_duration"**         on the y-axis. The data for these variables is taken from the pandas DataFrame **"df"**.
  * **plt.title('Time From Last Drug Taken VS Wear Off Duration')**
      This line sets the title of the plot to **"Time From Last Drug Taken VS Wear Off Duration".**
  * **plt.xlabel('Time From Last Drug Taken')**
      This line sets the label for the x-axis to **"Time From Last Drug Taken".**
  * **plt.ylabel('Wear Off Duration')**
      This line sets the label for the y-axis to **"Wear Off Duration".**
  * **plt.grid(True)**
      This line adds a grid to the plot.
  * **plt.savefig("DrugTime.jpg")**
      This line saves the plot as an image file named **"DrugTime.jpg".**
  * **plt.show()**
      This line displays the plot on the screen.

   Same as before, This plot can be used to visualize the relationship between the two variables and to identify any patterns or trends in the data.

* **★ Set IV and DV ★**
  &nbsp;&nbsp;
  ```
  x = df[['heart_rate', 'stress_score', 'time_from_last_drug_taken']]
  y = df['wo_duration']
  ```
  The code is used to select a subset of columns from a pandas DataFrame called **"df"** and store them in a new DataFrame called **"x"**. The selected columns are **"heart_rate", "stress_score", and "time_from_last_drug_taken".** The **"wo_duration"** column is also selected and stored in a separate variable called **"y"**. 

  And also this code is used to select a subset of columns from a pandas DataFrame and store them in a new DataFrame, as well as to select a single column and store it in a separate variable. 

  Here is a summary of what each line of the code does:
  * **x = df[['heart_rate', 'stress_score', 'time_from_last_drug_taken']]**
      This line creates a new DataFrame called **"x"** that contains the columns **"heart_rate", "stress_score",        and "time_from_last_drug_taken"** from the pandas DataFrame **"df".**
  * **y = df['wo_duration']**
      This line creates a new variable called **"y"** that contains the **"wo_duration" column from the pandas          DataFrame "df".**

* **★ Calculate Regression with Sklearn ★**
 &nbsp;&nbsp;
```
regr = linear_model.LinearRegression()
regr.fit(x, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
```
 
The function of the given code is to perform linear regression using the **`LinearRegression()`** model from the **`linear_model`** module in the scikit-learn library.
Here is an explanation of each line of code:
* **`regr = linear_model.LinearRegression()`** : Creates a `regr` object which is an instance of the `LinearRegression()` class. This object will be used to perform linear regression.
* **`regr.fit(x, y)`** : Uses the `fit()` method on the `regr` object to train a linear regression model. `x` is input data which contains the features used to make predictions, while `y` is the target or value to be predicted.
* **`print('Intercept: \n', regr. intercept_)`** : Returns the intercept value of the linear regression model. The intercept value is a constant value in the linear regression equation.
* **`print('Coefficients: \n', regr.coef_)`** : Returns the coefficients of the linear regression model. The coefficient value describes the relationship between each feature and the target to be predicted.

Using the above code, you can train a linear regression model with **`x`** and **`y`** data, and then display the intercept values and coefficients obtained from the model.

* **★ Calculate Regression with Statsmodels ★**
 &nbsp;&nbsp;
```
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
print_model = model.summary()
print(print_model)
```
The given code has some functionality related to linear regression using the statsmodels library. Here's an explanation of each line of code:
* **`x = sm.add_constant(x)`** : Adds a constant to the **`x`** input data. The **`add_constant()`** method from `statsmodels` is used to add constant columns that are normally needed in linear regression models. This constant represents the intercept or bias value in the regression equation.

* **`model = sm.OLS(y, x).fit()`** : Creates a linear regression model object using the Ordinary Least Squares (OLS) method of `statsmodels`. **`y`** is the target or value to be predicted, and `x` is the feature matrix including the added constants. The **`fit()`** method is used to train the model with the given data.

* **`predictions = model.predict(x)`** : Uses a trained model to make predictions on `x` data. The **`predict()`** method is used to generate predictions based on a trained linear regression model.

* **`print_model = model.summary()`** : Gets the summary statistics of the trained linear regression model. The **`summary()`** method is used to generate a summary that includes various important statistics such as coefficient values, p-values, R-squared, etc.

* **`print(print_model)`** : Prints the summary statistics of the model to output.
  
Using the code, you can train linear regression models, make predictions, and get summary statistics from the model using the statsmodels library.

* **★ Residual Plot Line Model ★**
 &nbsp;&nbsp;
```
from sklearn.linear_model import LinearRegression
x = df['heart_rate']
```
* The code given imports the LinearRegression class from the **linear_model** module in the scikit-learn library. Next, x is initialized with data from the **'heart_rate'** column of a DataFrame df.
  

&nbsp;&nbsp;
```
model = LinearRegression()
```
The function and purpose of the **model = LinearRegression()** line of code is as follows:
* **Function**:
1. Creates a linear regression model object using the LinearRegression class from the linear_model module in the scikit-learn library.
2. This model object will be used to train and make predictions using the linear regression method.

* **Objective**:
1. Initialize the model object that will be used to perform linear regression on the data.
2. Allows us to train models on training data and use them to make predictions on new data.
3. The resulting linear regression model can be used to study the relationship between input variables (features) and target variables (values to be predicted).


&nbsp;&nbsp;
```
model.fit(x.values.reshape(-1, 1), y)
```
The function of the **`model.fit(x.values.reshape(-1, 1), y)`** line of code is to train a linear regression model using training data **`x`** and target values **`y`**.

The following is an explanation of each component in the code:
1. **`x.values.reshape(-1, 1)`**: Takes the value from the `x` variable and converts it into a one-dimensional array using **`.values`**. Then, use **`.reshape(-1, 1)`** to convert the array into a matrix with dimensions suitable for use in the linear regression model.
- Reshaping is done with **`-1`** : the number of rows is adjusted automatically based on the second dimension specified, i.e. `1`.
- This is necessary because the linear regression model in scikit-learn expects a feature matrix with two dimensions.

2. **`y`**: The `y` variable contains the target value to be predicted. This target value must match the training data used.

3. **`model.fit(x.values.reshape(-1, 1), y)`**: The **`.fit()`** method on the `model` object is used to train the linear regression model. The training data `x` and target values `y` are used as arguments and the model learns the relationship between the input and target variables.

After calling the **`.fit()`** method, the linear regression model will be trained using the training data provided, so that it can be used to make predictions on new data.


&nbsp;&nbsp;
```
model.intercept_
```


&nbsp;&nbsp;
```
model.coef_
```


&nbsp;&nbsp;
```
# Create a figure and axes for the plot
figure = plt.figure(figsize=(10, 5))
axes = plt.axes()

# Display the data points using a scatterplot
sns.scatterplot(x=x, y=y, ax=axes)

# Plot the regression line
sns.lineplot(x=[0, 10], y=[model.intercept_, (10 * model.coef_[0] + model.intercept_)], ax=axes, color='blue')

# Plot the residuals
for x_value, y_value in zip(x, y):
    predicted_y = x_value * model.coef_[0] + model.intercept_
    axes.plot([x_value, x_value], [y_value, predicted_y], color='red')

# Save the plot as an image file named 'HeartRateLine.jpg'
plt.savefig('HeartRateLine.jpg')
```

  
###### HEADING 4

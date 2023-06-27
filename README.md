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
###### HEADING 4

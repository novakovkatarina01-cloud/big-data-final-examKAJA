# Big Data Final Exam – Customer Behaviour Analysis from Keggle
Author: **Katarina Novakov** :)

- Project Overview
This project analyses customer behaviour using PySpark and SQL inside Databricks.  
It includes data ingestion, cleaning, transformation, exploratory analysis, SQL analytics, dashboard creation, and an optional machine learning component for bonus marks.



#Contents of This Repository
This repository contains:

### ✔ Notebooks
- `python_behavior_notebook.ipynb` — PySpark data cleaning, transformation, ML model training, saving model  
- `sql_behavior_notebook.ipynb` — SQL analytical queries used for insights and dashboards

### ✔ Data
- `customer_behaviour.csv`

### ✔ Dashboard Screenshots
- `sql editor for season dashboard.png`
- `sql editor for age dashboard.png`
- `customer behavior dashboard.png`





##  Technologies Used
- **Databricks (Free Edition)**
- **PySpark**
- **Spark SQL**
- **Delta Tables**
- **Databricks SQL Dashboards**
- **PySpark MLlib (Linear Regression Model)**



## Data Pipeline Steps

### **Data Cleaning & Preparation (PySpark)**
Performed:
- Column renaming  
- Type casting  
- Handling missing values  
- Creating new columns (`age_group`)  
- Saving cleaned dataset as a Delta table:  
  `project.customer_behaviour_final`



### **SQL Analysis**
Two required SQL analytical queries were created:

####  Average spend by season  
```sql
SELECT season, ROUND(AVG(purchase_amount_usd),2) AS avg_spend
FROM project.customer_behaviour_final
GROUP BY season
ORDER BY avg_spend DESC;
and 
SELECT age_group, ROUND(AVG(purchase_amount_usd),2) AS avg_spend
FROM project.customer_behaviour_final
GROUP BY age_group
ORDER BY avg_spend DESC;



##  **Bonus Component — Machine Learning (PySpark MLlib)** 

To earn additional marks, a Machine Learning model was implemented using **PySpark MLlib**.  
The goal was to predict **purchase_amount_usd** based on other numerical features.

### **Steps Performed**

#### **1️ Feature Selection**
Selected numerical columns:
- `age`
- `previous_purchases`
- `review_rating`

#### **VectorAssembler**
All numerical features were combined into a single ML vector column:

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["age", "previous_purchases", "review_rating"],
    outputCol="features"
)

data_ml = assembler.transform(df_final)

train, test = data_ml.randomSplit([0.8, 0.2], seed=42)

###Train Linear Regression Model
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="purchase_amount_usd")
lr_model = lr.fit(train)
 ## And model evaluation
test_results = lr_model.evaluate(test)
rmse = test_results.rootMeanSquaredError
print("RMSE:", rmse)

##### Finally loading and data saved

from pyspark.ml.regression import LinearRegressionModel

loaded_model = LinearRegressionModel.load(model_path)




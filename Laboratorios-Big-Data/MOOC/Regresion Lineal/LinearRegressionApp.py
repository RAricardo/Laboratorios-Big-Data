# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("LinearRegApp").getOrCreate()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

data = spark.read.csv("/FileStore/tables/Ecommerce_Customers.csv", inferSchema = True, header = True)

# COMMAND ----------

data.show()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.head(1)

# COMMAND ----------

for item in data.head(2)[0]:
  print(item)

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols=['Avg Session Length','Time on App','Time on Website','Length of Membership','Yearly Amount Spent'], outputCol="features")

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

output.show()

# COMMAND ----------

final_data = output.select('features', 'Yearly Amount Spent')

# COMMAND ----------

final_data.show()

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7,0.3])

# COMMAND ----------

train_data.describe().show()

# COMMAND ----------

test_data.describe().show()

# COMMAND ----------

lr = LinearRegression(labelCol='Yearly Amount Spent')

# COMMAND ----------

lr_model = lr.fit(train_data)

# COMMAND ----------

test_result = lr_model.evaluate(test_data)

# COMMAND ----------

test_result.residuals.show()

# COMMAND ----------

test_result.rootMeanSquaredError

# COMMAND ----------

test_result.r2

# COMMAND ----------

final_data.describe().show()

# COMMAND ----------

unlabeled_data = test_data.select('features')

# COMMAND ----------

unlabeled_data.show()

# COMMAND ----------

predictions = lr_model.transform(unlabeled_data)

# COMMAND ----------

predictions.show()

# COMMAND ----------



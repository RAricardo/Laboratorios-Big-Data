# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("TreeMethodsProject").getOrCreate()

# COMMAND ----------

data = spark.read.csv("/FileStore/tables/dog_food.csv", inferSchema=True, header=True)

# COMMAND ----------

data.show()

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

assembler = VectorAssembler(inputCols=['A','B','C','D'], outputCol="features")

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

rfc = RandomForestClassifier(labelCol="Spoiled", featuresCol="features")

# COMMAND ----------

output.printSchema()

# COMMAND ----------

final_data = output.select("features", "Spoiled")

# COMMAND ----------

final_data.show()

# COMMAND ----------

rfc_model = rfc.fit(final_data)

# COMMAND ----------

final_data.head(1)

# COMMAND ----------

rfc_model.featureImportances

# COMMAND ----------



# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("KMeans").getOrCreate()

# COMMAND ----------

data = spark.read.csv("/FileStore/tables/seeds_dataset.csv", inferSchema=True, header=True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

assembler = VectorAssembler(inputCols=data.columns, outputCol="features")

# COMMAND ----------

final_data = assembler.transform(data)

# COMMAND ----------

final_data.printSchema()

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# COMMAND ----------

scaler = StandardScaler(inputCol="features", outputCol="Scaled Features")

# COMMAND ----------

scaler_model = scaler.fit(final_data) 

# COMMAND ----------

final_data = scaler_model.transform(final_data)

# COMMAND ----------

final_data.show()

# COMMAND ----------

kmeans= KMeans(featuresCol = "Scaled Features", k=3)

# COMMAND ----------

model = kmeans.fit(final_data)

# COMMAND ----------

print("WSSSE")
print(model.computeCost(final_data))

# COMMAND ----------

centers = model.clusterCenters()

# COMMAND ----------

print(centers)

# COMMAND ----------

model.transform(final_data).select("prediction").show()

# COMMAND ----------



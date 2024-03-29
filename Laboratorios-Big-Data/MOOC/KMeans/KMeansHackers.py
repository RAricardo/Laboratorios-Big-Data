# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("Kmeans").getOrCreate()

# COMMAND ----------

data = spark.read.csv("/FileStore/tables/hack_data.csv", inferSchema=True, header=True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols=['Session_Connection_Time',
 'Bytes Transferred',
 'Kali_Trace_Used',
 'Servers_Corrupted',
 'Pages_Corrupted',
 'WPM_Typing_Speed'], outputCol="features")

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

cluster_final_data = scaler_model.transform(final_data)

# COMMAND ----------

kmeans2 = KMeans(featuresCol="Scaled Features", k=2)

# COMMAND ----------

kmeans3 = KMeans(featuresCol="Scaled Features", k=3)

# COMMAND ----------

model_k2 = kmeans2.fit(cluster_final_data)
model_k3 = kmeans3.fit(cluster_final_data)

# COMMAND ----------

model_k3.transform(cluster_final_data).groupBy("prediction").count().show()

# COMMAND ----------

model_k2.transform(cluster_final_data).groupBy("prediction").count().show()

# COMMAND ----------



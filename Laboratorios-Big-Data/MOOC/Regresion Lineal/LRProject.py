# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("LRproject").getOrCreate()

# COMMAND ----------

df = sqlContext.sql("select * from cruise_ship_info_csv")

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

for ship in df.head(5):
  print(ship)
  print("\n")

# COMMAND ----------

df.groupBy("Cruise_line").count().show()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")

# COMMAND ----------

indexed = indexer.fit(df).transform(df)

# COMMAND ----------

indexed.head(1)

# COMMAND ----------

from pyspark.ml.linalg import Vectors

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

indexed.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols=['Age','Tonnage','passengers','length','cabins','passenger_density','cruise_cat'], outputCol="features")

# COMMAND ----------

output = assembler.transform(indexed)

# COMMAND ----------

output.select("features", "crew").show()

# COMMAND ----------

final_data = output.select(["features","crew"])

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7,0.3])

# COMMAND ----------

train_data.describe().show()

# COMMAND ----------

test_data.describe().show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

ship_lr = LinearRegression(labelCol="crew")

# COMMAND ----------

trained_ship_model = ship_lr.fit(train_data)

# COMMAND ----------

ship_results = trained_ship_model.evaluate(test_data)

# COMMAND ----------

ship_results.rootMeanSquaredError
ship_results.r2

# COMMAND ----------

train_data.describe().show()

# COMMAND ----------

from pyspark.sql.functions import corr

# COMMAND ----------

df.select(corr("crew","cabins")).show()

# COMMAND ----------



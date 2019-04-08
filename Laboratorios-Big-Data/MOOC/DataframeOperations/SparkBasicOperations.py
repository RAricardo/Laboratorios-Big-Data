# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('ops').getOrCreate()

# COMMAND ----------

df = sqlContext.sql("select * from appl_stock_csv")

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.filter("Close < 500").select(["Open", "Close"]).show()

# COMMAND ----------

df.filter((df["Close"] < 200) & (df["Open"] >200)).show()

# COMMAND ----------

result = df.filter(df["Low"] == 197.16).collect()

# COMMAND ----------

result

# COMMAND ----------

row = result[0]

# COMMAND ----------

row.asDict()["Volume"]

# COMMAND ----------



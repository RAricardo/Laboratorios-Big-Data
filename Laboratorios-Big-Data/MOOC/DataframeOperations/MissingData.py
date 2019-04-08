# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("miss").getOrCreate()

# COMMAND ----------

df = sqlContext.sql("select * from containsnull_csv")

# COMMAND ----------

df.show()

# COMMAND ----------

df.na.drop().show()

# COMMAND ----------

df.na.drop(thresh=2).show()

# COMMAND ----------

df.na.drop(how="all").show()

# COMMAND ----------

df.na.drop(subset="Sales").show()

# COMMAND ----------

df.na.fill('No Name', subset=["Name"]).show()

# COMMAND ----------

from pyspark.sql.functions import mean

# COMMAND ----------

mean_val = df.select(mean(df["Sales"])).collect()

# COMMAND ----------

mean_val[0]

# COMMAND ----------

mean_sales = mean_val[0][0]

# COMMAND ----------

mean_sales

# COMMAND ----------

df.na.fill(mean_sales, ["Sales"]).show()

# COMMAND ----------

df.na.fill(df.select(mean(df["Sales"])).collect()[0][0],["Sales"]).show()

# COMMAND ----------



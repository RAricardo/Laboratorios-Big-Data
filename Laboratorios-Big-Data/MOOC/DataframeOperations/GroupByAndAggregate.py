# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("aggs").getOrCreate()

# COMMAND ----------

df = sqlContext.sql("select * from sales_info_csv")

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.groupBy("Company")

# COMMAND ----------

df.groupBy("Company").count().show()

# COMMAND ----------

df.agg({'Sales':'sum'}).show()

# COMMAND ----------

group_data = df.groupBy("Company")

# COMMAND ----------

group_data.agg({"Sales": "max"}).show()

# COMMAND ----------

from pyspark.sql.functions import countDistinct, avg, stddev

# COMMAND ----------

df.select(stddev("Sales").alias("Average Sales")).show()

# COMMAND ----------

from pyspark.sql.functions import format_number

# COMMAND ----------

sales_std = df.select(stddev("Sales").alias("std"))

# COMMAND ----------

sales_std.show()

# COMMAND ----------

sales_std.select(format_number("std", 2).alias("STD")).show()

# COMMAND ----------

df.show()

# COMMAND ----------

df.orderBy(df["Sales"].desc()).show()

# COMMAND ----------



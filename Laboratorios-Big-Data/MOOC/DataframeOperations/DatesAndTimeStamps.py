# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("dates").getOrCreate()

# COMMAND ----------

df = sqlContext.sql("select * from appl_stock_csv")

# COMMAND ----------

df.head(1)

# COMMAND ----------

from pyspark.sql.functions import (dayofmonth, hour, dayofyear,
                                   month, year, weekofyear,
                                   format_number, date_format)

# COMMAND ----------

df.select(dayofmonth(df["Date"])).show()

# COMMAND ----------

df.select(year(df["Date"])).show()

# COMMAND ----------

df.withColumn("Year",year(df["Date"])).show()

# COMMAND ----------

newdf = df.withColumn("Year",year(df["Date"]))

# COMMAND ----------

newdf.groupBy("Year").mean().show()

# COMMAND ----------

result = newdf.groupBy("Year").mean().select(["Year", "avg(Close)"])

# COMMAND ----------

new = result.withColumnRenamed("avg(Close)", "Average Closing Price")

# COMMAND ----------

new.select(["Year", format_number("Average Closing Price", 2).alias("Average Close")]).show()

# COMMAND ----------



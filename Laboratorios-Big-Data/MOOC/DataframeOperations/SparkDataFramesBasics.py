# Databricks notebook source
import pyspark

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("Basics").getOrCreate()

# COMMAND ----------

df = sqlContext.sql("SELECT * FROM people_json")

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.columns

# COMMAND ----------

df.describe().show()

# COMMAND ----------

from pyspark.sql.types import (StructField, StringType,
                               IntegerType, StructType)

# COMMAND ----------

data_schema = [StructField('age', IntegerType(), True),
               StructField('name', StringType(), True)]

# COMMAND ----------

final_struc = StructType(fields = data_schema)

# COMMAND ----------

df = spark.read.table("people_json")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.show()

# COMMAND ----------

df['age']

# COMMAND ----------

df.select('age').show()

# COMMAND ----------

df.head(2)

# COMMAND ----------

df.withColumn("double_Age", df["age"]*2).show()

# COMMAND ----------



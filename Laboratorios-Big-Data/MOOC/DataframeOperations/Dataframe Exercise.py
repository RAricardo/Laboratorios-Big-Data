# Databricks notebook source
#Start a Simple Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("exercise").getOrCreate()

# COMMAND ----------

#Load Walmart Stock files, have spark infer the data types
df = sqlContext.sql("select * from walmart_stock_csv")

# COMMAND ----------

#Print table
df.show()

# COMMAND ----------

#List Column Names
df.columns

# COMMAND ----------

#What does the Schema looks like?
df.printSchema()

# COMMAND ----------

#Print first 5 columns
df.head(5)

# COMMAND ----------

#Use describe to print about the dataframe
df.describe().show()

# COMMAND ----------

#Format mean and stddev in describe() to only show to decimal places
from pyspark.sql.functions import format_number

# COMMAND ----------

result = df.describe()

# COMMAND ----------

result.select(result["summary"], format_number(result["Open"].cast("float"), 2).alias("Open")).show()

# COMMAND ----------

#Add a column representing the ratio of High vs Volume (High/Volume)
df2 = df.withColumn("HV RATIO", df["High"]/df["Volume"]).show()

# COMMAND ----------

#What day had the highest peak price
df.orderBy(df["High"].desc()).head(1)[0][0]

# COMMAND ----------

#What is the mean of the Close Column
from pyspark.sql.functions import mean
df.select(mean(df["Close"])).show()

# COMMAND ----------

#What is the max and the min of the volume column
from pyspark.sql.functions import min, max
df.select(min(df["Volume"]).alias("Min Volume"), max(df["Volume"]).alias("Max Volume") ).show()

# COMMAND ----------

#How many days was the close lower than 60
df.filter(df["Close"] < 60).count()

# COMMAND ----------

#What percentage of the time was the High greater than 80.
(df.filter(df["High"]>80).count() / df.count())*100

# COMMAND ----------

#What is the Pearson Correlation between High and Volumne
from pyspark.sql.functions import corr
df.select(corr(df["High"], df["Volume"])).show()

# COMMAND ----------

#What is the max High per Year
from pyspark.sql.functions import year
yeardf = df.withColumn("Year", year(df["Date"]))
max_df = yeardf.groupBy("Year").max()
max_df.select("Year", "max(High)").show()

# COMMAND ----------



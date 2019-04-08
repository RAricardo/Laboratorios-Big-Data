# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("RS").getOrCreate()

# COMMAND ----------

from pyspark.ml.recommendation import ALS

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

data = spark.read.csv("/FileStore/tables/movielens_ratings.csv", inferSchema=True, header=True)

# COMMAND ----------

data.show()

# COMMAND ----------

data.describe().show()

# COMMAND ----------

training,test = data.randomSplit([0.7,0.3])

# COMMAND ----------

als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')

# COMMAND ----------

model = als.fit(training)

# COMMAND ----------

predictions = model.transform(test)

# COMMAND ----------

predictions.show()

# COMMAND ----------

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# COMMAND ----------

rmse = evaluator.evaluate(predictions)

# COMMAND ----------

print("RSME")
print(rmse)

# COMMAND ----------

single_user = test.filter(test["userId"]==11).select(["movieId", "userId"])

# COMMAND ----------

single_user.show()

# COMMAND ----------

recommendations = model.transform(single_user)

# COMMAND ----------

recommendations.orderBy('prediction', ascending=False).show()

# COMMAND ----------



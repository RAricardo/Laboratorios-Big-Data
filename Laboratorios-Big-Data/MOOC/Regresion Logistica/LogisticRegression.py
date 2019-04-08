# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("logreg").getOrCreate()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

my_data = spark.read.format("libsvm").load("/FileStore/tables/sample_libsvm_data.txt")

# COMMAND ----------

my_data.show()

# COMMAND ----------

my_log_reg_model = LogisticRegression()

# COMMAND ----------

fitted_logreg = my_log_reg_model.fit(my_data)

# COMMAND ----------

log_summary = fitted_logreg.summary

# COMMAND ----------

log_summary.predictions.printSchema()

# COMMAND ----------

log_summary.predictions.show()

# COMMAND ----------

training_data, test_data = my_data.randomSplit([0.7,0.3])

# COMMAND ----------

final_model = LogisticRegression()

# COMMAND ----------

fit_final = final_model.fit(training_data)

# COMMAND ----------

predictions_and_labels = fit_final.evaluate(test_data)

# COMMAND ----------

predictions_and_labels.predictions.show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

my_eval = BinaryClassificationEvaluator()

# COMMAND ----------

my_final_roc = my_eval.evaluate(predictions_and_labels.predictions)

# COMMAND ----------

my_final_roc

# COMMAND ----------



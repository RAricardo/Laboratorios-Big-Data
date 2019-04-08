# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("TreeMethods").getOrCreate()

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, DecisionTreeClassifier

# COMMAND ----------

data = spark.read.format("libsvm").load("/FileStore/tables/sample_libsvm_data.txt")

# COMMAND ----------

data.show()

# COMMAND ----------

training_set, test_data = data.randomSplit([0.7,0.3])

# COMMAND ----------

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier(numTrees=100)
gbt = GBTClassifier()

# COMMAND ----------

dtc_model=dtc.fit(training_set)
rfc_model=rfc.fit(training_set)
gbt_model=gbt.fit(training_set)

# COMMAND ----------

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)

# COMMAND ----------

dtc_preds.show()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")

# COMMAND ----------

print("DTC Accuracy:")
acc_eval.evaluate(dtc_preds)

# COMMAND ----------

rfc_model.featureImportances

# COMMAND ----------



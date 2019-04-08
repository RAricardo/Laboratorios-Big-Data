# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("TreeMethodsCollege").getOrCreate()

# COMMAND ----------

data = spark.read.csv("/FileStore/tables/College.csv", inferSchema=True, header=True)

# COMMAND ----------

data.show()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols=['Apps',
 'Accept',
 'Enroll',
 'Top10perc',
 'Top25perc',
 'F_Undergrad',
 'P_Undergrad',
 'Outstate',
 'Room_Board',
 'Books',
 'Personal',
 'PhD',
 'Terminal',
 'S_F_Ratio',
 'perc_alumni',
 'Expend',
 'Grad_Rate'],outputCol="features")

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol="Private", outputCol="PrivateIndex")

# COMMAND ----------

output_fixed = indexer.fit(output).transform(output)

# COMMAND ----------

output_fixed.printSchema()

# COMMAND ----------

my_final_data = output_fixed.select("features", "PrivateIndex")

# COMMAND ----------

training_data, test_data = my_final_data.randomSplit([0.7,0.3])

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

dtc = DecisionTreeClassifier(labelCol="PrivateIndex", featuresCol="features")
rfc = RandomForestClassifier(numTrees=150, labelCol="PrivateIndex", featuresCol="features")
gbt = GBTClassifier(labelCol="PrivateIndex", featuresCol="features")

# COMMAND ----------

dtc_model = dtc.fit(training_data)
rfc_model = rfc.fit(training_data)
gbt_model = gbt.fit(training_data)

# COMMAND ----------

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

my_binary_eval = BinaryClassificationEvaluator(labelCol="PrivateIndex")

# COMMAND ----------

print("DTC")
print(my_binary_eval.evaluate(dtc_preds))
print("RFC")
print(my_binary_eval.evaluate(rfc_preds))
print("GBT")
print(my_binary_eval.evaluate(gbt_preds))

# COMMAND ----------



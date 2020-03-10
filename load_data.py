from pyspark.sql import SparkSession, functions as F, types, Row
cluster_seeds = ['127.0.0.1']
spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext


def load_patients():
    df = spark.read.format("csv").option("header", "true").load("PATIENTS.csv.gz")
    df = df.select("SUBJECT_ID","DOB")
    df = df.withColumnRenamed("SUBJECT_ID", "subject_id")
    df = df.withColumnRenamed("DOB", "dob")
    df

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH PATIENTS.")

def load_admissions():
    df = spark.read.format("csv").option("header", "true").load("ADMISSIONS.csv.gz")
    df = df.select("SUBJECT_ID","HADM_ID","ADMITTIME","ADMISSION_TYPE")
    df = df.withColumnRenamed("SUBJECT_ID", "subject_id")
    df = df.withColumnRenamed("HADM_ID", "hadm_id")
    df = df.withColumnRenamed("ADMITTIME", "admittime")
    df = df.withColumnRenamed("ADMISSION_TYPE", "admission_type")
    df.show()
    df.write.format("org.apache.spark.sql.cassandra").options(table='admissions', keyspace='mimic').save()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH ADMISSIONS.")


def load_labelitems(item_n):
    event_schema = types.StructType([
        types.StructField('row_id', types.IntegerType()),
        types.StructField('subject_id', types.IntegerType()),
        types.StructField('hadm_id', types.IntegerType()),
        types.StructField('itemid', types.IntegerType()),
        types.StructField('charttime', types.TimestampType()),
        types.StructField('value', types.StringType()),
        types.StructField('valuenum', types.FloatType()),
        types.StructField('valueuom', types.StringType()),
        types.StructField('flag', types.BooleanType()),
    ])

    df = spark.read.csv("LABEVENTS.csv.gz", schema = event_schema)
    df = df[df.itemid == item_n]
    df = df.select('row_id','hadm_id','subject_id','charttime', 'valuenum','valueuom')
    df = df.filter(df.hadm_id.isNotNull() & df.subject_id.isNotNull() & df.valuenum.isNotNull())
    print(df.schema)
    df.write.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').save()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH LABELITEMS.")


def check_valueuom(item_n):
    df = spark.read.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').load()
    total_n = df.count()
    null_n = df.filter(df.valueuom.isNull()).count()
    majority_n = df[df.valueuom == "%"].count()
    print("Number of rows: "+str(total_n))
    print("Number of rows whose valueuom is null: "+str(null_n))
    print("Number of rows whose valueuom is %: "+str(majority_n))
    print("Number of rows whose valueuom is not null and not %: "+str(total_n - null_n - majority_n))


def save_first_record(item_n):
    df = spark.read.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').load()
    df_minct = df.groupby(["hadm_id","subject_id"]).agg(F.min(df["charttime"]))
    df_minct = df_minct.withColumnRenamed("min(charttime)", "charttime")
    df_minct.show()
    df_result = df_minct.join(df, ["hadm_id","subject_id","charttime"]).select(["hadm_id","subject_id","charttime","valuenum"])
    df_result.show()
    df_result.write.format("org.apache.spark.sql.cassandra").options(table="item"+str(item_n), keyspace='mimic').save()


def load_chartevents():
    event_schema = types.StructType([
        types.StructField('row_id', types.IntegerType()),
        types.StructField('subject_id', types.IntegerType()),
        types.StructField('hadm_id', types.IntegerType()),
        types.StructField('icustay_id', types.IntegerType()),
        types.StructField('itemid', types.IntegerType()),
        types.StructField('charttime', types.TimestampType()),
        types.StructField('storetime', types.TimestampType()),
        types.StructField('cgid', types.IntegerType()),
        types.StructField('value', types.StringType()),
        types.StructField('valuenum', types.FloatType()),
        types.StructField('valueuom', types.StringType()),
        types.StructField('warning', types.IntegerType()),
        types.StructField('error', types.IntegerType()),
        types.StructField('resultstatus', types.StringType()),
        types.StructField('stopped', types.StringType()),
    ])

    df = spark.read.csv("CHARTEVENTS.csv.gz", schema = event_schema)

    ce_itemids = [723, 454, 184, 223900, 223901, 220739,
               51, 442, 455, 6701, 220179, 220050,
               211, 220045, 
               678, 223761, 676, 223762,
               223835, 3420, 3422, 190]
    for itemid in ce_itemids:
        df = df.where(df.itemid == itemid)
        df.write.format("org.apache.spark.sql.cassandra").options(table=str(itemid), keyspace='mimic').save()
        print("DONE WITH ITEM "+str(itemid)+" IN CHARTEVENTS")


if __name__== "__main__":
  #load_patients()
  #load_admissions()
  item_n = 50816
  #load_labelitems(item_n)
  #check_valueuom(item_n)
  save_first_record(item_n)

"""
le_itemids = [50821, 50816,
              51006,
              51300,51301,
              50882, 
              950824, 50983,
              50822, 50971,
              50885]

oe_itemids = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 
              40651, 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
"""
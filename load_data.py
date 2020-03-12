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

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH PATIENTS.")

def load_admissions():
    adm_schema = types.StructType([
        types.StructField('row_id', types.IntegerType()),
        types.StructField('subject_id', types.IntegerType()),
        types.StructField('hadm_id', types.IntegerType()),
        types.StructField('admittime', types.TimestampType()),
        types.StructField('dischtime', types.TimestampType()),
        types.StructField('deathtime', types.TimestampType()),
        types.StructField('admission_type', types.StringType()),
        types.StructField('admission_location', types.StringType()),
        types.StructField('discharge_location', types.StringType()),
        types.StructField('insurance', types.StringType()),
        types.StructField('language', types.StringType()),
        types.StructField('religion', types.StringType()),
        types.StructField('marital_status', types.StringType()),
        types.StructField('ethnicity', types.StringType()),
        types.StructField('edregtimen', types.TimestampType()),
        types.StructField('edouttimen', types.TimestampType()),
        types.StructField('diagnosis', types.StringType()),
        types.StructField('hospital_expire_flag', types.IntegerType()),
        types.StructField('has_chartevents_flag', types.IntegerType()),
    ])
    df = spark.read.csv("ADMISSIONS.csv.gz", schema = adm_schema)
    df = df.select("subject_id","hadm_id","admittime","dischtime","admission_type","hospital_expire_flag")
    df = df[(df.hadm_id.isNotNull())&(df.subject_id.isNotNull())]
    df.show()
    df.write.format("org.apache.spark.sql.cassandra").options(table='admissions', keyspace='mimic').save()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH ADMISSIONS.")


def load_labitems(item_n):
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
    df.show()
    df.write.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').save()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH LABELITEMS.")

def load_outputitems(item_n1, item_n2):
    event_schema = types.StructType([
        types.StructField('row_id', types.IntegerType()),
        types.StructField('subject_id', types.IntegerType()),
        types.StructField('hadm_id', types.IntegerType()),
        types.StructField('icustay_id', types.IntegerType()),
        types.StructField('charttime', types.TimestampType()),
        types.StructField('itemid', types.IntegerType()),
        types.StructField('value', types.DoubleType()),
        types.StructField('valueuom', types.StringType()),
        types.StructField('storetime', types.TimestampType()),
        types.StructField('cgid', types.LongType()),
        types.StructField('stopped', types.StringType()),
        types.StructField('newbottle', types.IntegerType()),
        types.StructField('iserror', types.ShortType()),
    ])

    df = spark.read.csv("OUTPUTEVENTS.csv.gz", schema = event_schema)
    df = df[(df.itemid == item_n1) | (df.itemid == item_n2)]
    df = df.select('row_id','hadm_id','subject_id','charttime', 'value','valueuom')
    df = df.withColumnRenamed("value", "valuenum")
    df = df.filter(df.hadm_id.isNotNull() & df.subject_id.isNotNull() & df.valuenum.isNotNull())
    print(df.schema)
    df.show()
    df.write.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n2), keyspace='mimic').save()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH OUTPUTLITEMS.")


def check_valueuom(item_n, uom):
    df = spark.read.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').load()
    total_n = df.count()
    null_n = df.filter(df.valueuom.isNull()).count()
    majority_n = df[(df.valueuom == uom)].count()
    others = df[(df.valueuom.isNotNull()) & (df.valueuom != uom)].show()
    print("Number of rows: "+str(total_n))
    print("Number of rows whose valueuom is null: "+str(null_n))
    print("Number of rows whose valueuom is "+uom+" or Deg. F: "+str(majority_n))
    print("Number of rows whose valueuom is not null and not "+uom+": "+str(total_n - null_n - majority_n))


def save_first_record(item_n):
    df = spark.read.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').load()
    df = df[(df.itemid == 51) | (df.itemid == 220050)]
    df_min_ct = df.groupby(["hadm_id","subject_id"]).agg(F.min(df["charttime"]))
    df_min_ct = df_min_ct.withColumnRenamed("min(charttime)", "charttime")
    df_min_ct.show()
    df_result = df_min_ct.join(df, ["hadm_id","subject_id","charttime"]).select(["hadm_id","subject_id","charttime","valuenum","itemid"])
    df_result.show()
    df_result.write.format("org.apache.spark.sql.cassandra").options(table="item"+str(item_n), keyspace='mimic').save()


def show_diff(item_n, num1, num2):
    df = spark.read.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n), keyspace='mimic').load()
    #df = df[(df.valueuom == uom) | (df.valueuom.isNull())]
    #print(df.count())
    df1 = df[(df.itemid == num1)]
    df2 = df[(df.itemid == num2)]
    df1 = df1.groupby(["hadm_id","subject_id"]).agg(F.min(df1["charttime"])).withColumnRenamed("min(charttime)", "charttime").join(df1, ["hadm_id","subject_id","charttime"]).select(["hadm_id","subject_id","charttime","itemid","valuenum"])
    df2 = df2.groupby(["hadm_id","subject_id"]).agg(F.min(df2["charttime"])).withColumnRenamed("min(charttime)", "charttime").join(df2, ["hadm_id","subject_id","charttime"]).select(["hadm_id","subject_id","charttime","itemid","valuenum"])
    df_result = df1.join(df2, ["hadm_id"])
    df_result.show()
    df[(df.itemid == item_n)].groupby(["hadm_id","subject_id"]).agg(F.min(df["charttime"])).withColumnRenamed("min(charttime)", "charttime").join(df, ["hadm_id","subject_id","charttime"]).select(["hadm_id","subject_id","charttime","itemid","valuenum"]).show()

def load_chartitems(item_n1, item_n2, item_n3, item_n4):
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
    df = df[(df.itemid == item_n1) | (df.itemid == item_n2) | (df.itemid == item_n3) | (df.itemid == item_n4)]
    print(df.head(2))
    df = df.select('row_id','hadm_id','subject_id','charttime', 'valuenum','valueuom', 'itemid')
    df = df.filter(df.hadm_id.isNotNull() & df.subject_id.isNotNull() & df.valuenum.isNotNull())
    #df.show()
    df.write.format("org.apache.spark.sql.cassandra").options(table='temp'+str(item_n2), keyspace='mimic').save()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("DONE WITH CHARTITEMS.")



if __name__== "__main__":
  #load_patients()
  #load_admissions()
  #load_labitems(item_n)
  #load_outputitems(item_n1, item_n2)
  item_n1 = 51
  item_n2 = 220050
  item_n3 = 6701

  #uom = "%"
  #load_chartitems(item_n1, item_n2, item_n3, item_n4)
  #check_valueuom(item_n2, uom)
  #show_diff(item_n2, item_n1, item_n3)
  save_first_record(item_n2)

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
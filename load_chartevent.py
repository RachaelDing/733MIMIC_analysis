from pyspark.sql import SparkSession, functions as F, types, Row
cluster_seeds = ['127.0.0.1']
spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
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
#head_a = df.head(5)
#a = df[df.itemid == 224329]
#head_a = a.head(1)
#for h in head_a:
#    print(h)

df = df.where((df.itemid==723) | (df.itemid==454)|(df.itemid==184) | (df.itemid==223900)
             |(df.itemid==223901) | (df.itemid==220739)
             |(df.itemid==51) | (df.itemid==442)|(df.itemid==455) | (df.itemid==6701)
             |(df.itemid==220179) | (df.itemid==220050)
             |(df.itemid==221) | (df.itemid==220045)
             |(df.itemid==678) | (df.itemid==223761)|(df.itemid==676) | (df.itemid==223762)
             |(df.itemid==223835) | (df.itemid==3420)|(df.itemid==3422) | (df.itemid==190))
#df = df.where(df.itemid==723)
#print(df.count())
df.write.format("org.apache.spark.sql.cassandra").options(table="chartevent", keyspace='mimic').save()

print("DONE")

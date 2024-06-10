# Databricks notebook source
# MAGIC %md
# MAGIC #### Imports

# COMMAND ----------

# MAGIC %pip install "unstructured[pdf]" mlflow cachetools databricks-vectorsearch databricks-genai-inference
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up the Vector Database

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

#Create the Endpoint
VS_ENDPOINT_NAME = 'genai_endpoint'

if vsc.list_endpoints().get('endpoints') == None or not VS_ENDPOINT_NAME in [endpoint.get('name') for endpoint in vsc.list_endpoints().get('endpoints')]:
    print(f"Creating new Vector Search endpoint named {VS_ENDPOINT_NAME}")
    vsc.create_endpoint(VS_ENDPOINT_NAME)
else:
    print(f"Endpoint {VS_ENDPOINT_NAME} already exists.")

vsc.wait_for_endpoint(VS_ENDPOINT_NAME, 600)

# COMMAND ----------

CATALOG = "workspace"
DB='default'
SOURCE_TABLE_NAME = "source_table"
SOURCE_TABLE_FULLNAME=f"{CATALOG}.{DB}.{SOURCE_TABLE_NAME}"

# Create the Vector Index
VS_INDEX_NAME = 'dais_hackathon'
VS_INDEX_FULLNAME = f"{CATALOG}.{DB}.{VS_INDEX_NAME}"


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note that we need to enable Change Data Feed on the table to create the index

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE OR REPLACE TABLE workspace.default.source_table (
# MAGIC   element_id STRING,
# MAGIC   type STRING,
# MAGIC   text STRING,
# MAGIC   file STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create vector search index

# COMMAND ----------


if not VS_INDEX_FULLNAME in [index.get("name") for index in vsc.list_indexes(VS_ENDPOINT_NAME).get('vector_indexes', [])]:
    try:
        # set up an index with managed embeddings
        print("Creating Vector Index...")
        i = vsc.create_delta_sync_index_and_wait(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_FULLNAME,
            source_table_name=SOURCE_TABLE_FULLNAME,
            pipeline_type="TRIGGERED",
            primary_key="element_id",
            embedding_source_column="text",
            embedding_model_endpoint_name="databricks-bge-large-en"
        )
    except Exception as e:
        if "INTERNAL_ERROR" in str(e):
            # Check if the index exists after the error occurred
            if VS_INDEX_FULLNAME in [index.get("name") for index in vsc.list_indexes(VS_ENDPOINT_NAME).get('vector_indexes', [])]:
                print(f"Index {VS_INDEX_FULLNAME} has been created.")
            else:
                raise e
        else:
            raise e
else:
    print(f"Index {VS_INDEX_FULLNAME} already exists.")
     


# COMMAND ----------

# MAGIC %md
# MAGIC Getting our pdf files (saved in a volume)

# COMMAND ----------

all_file_names = [file.name for file in dbutils.fs.ls("/Volumes/workspace/default/climate_reports")]

# COMMAND ----------

from unstructured.partition.auto import partition
data_list = []
def process_file(file_name):
    # Specify the path to the PDF file
    pdf_file_path = f"/Volumes/workspace/default/climate_reports/{file_name}"

    elements = partition(pdf_file_path)

    # Process the extracted elements
    for element in elements:
        element_dict = element.to_dict()
        element_dict["file_name"] = file_name
        data_list.append(element_dict)
    

# COMMAND ----------

for file in all_file_names:
    process_file(file)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a dataframe with the exported chunks

# COMMAND ----------

import pyspark.sql.types as T

# Define schema
schema = T.StructType([
    T.StructField("type", T.StringType(), True),
    T.StructField("element_id", T.StringType(), True),
    T.StructField("text", T.StringType(), True),
    T.StructField("file", T.StringType(), True),
    T.StructField("metadata", T.MapType(T.StringType(), T.StringType()), True)
])

# Create DataFrame
df = spark.createDataFrame(data_list, schema=schema).drop("metadata")


# COMMAND ----------

# populate the source table
df.write.format("delta").mode("append").saveAsTable(
        SOURCE_TABLE_FULLNAME
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sync the Vector Search Index 

# COMMAND ----------

index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME,
                      index_name=VS_INDEX_FULLNAME)
index.sync()

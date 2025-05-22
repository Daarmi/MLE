import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window


from functools import reduce
from pyspark.sql import DataFrame


from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# Run Bronze Backfill - Updated for Assignment
for date_str in dates_str_lst:
    click_df  = utils.data_processing_bronze_table.process_clickstream_bronze_table(
                    date_str, bronze_lms_directory, spark)
    attr_df   = utils.data_processing_bronze_table.process_attributes_bronze_table(
                    date_str, bronze_lms_directory, spark)
    fin_df    = utils.data_processing_bronze_table.process_financials_bronze_table(
                    date_str, bronze_lms_directory, spark)
    loan_df   = utils.data_processing_bronze_table.process_loan_daily_bronze_table(
                    date_str, bronze_lms_directory, spark)


# create Silver datalake - Updated for Assignment
# ────────────────────────────────────────────────────────────────
#  SILVER back-fill  (one output folder per dataset)
# ────────────────────────────────────────────────────────────────
silver_clickstream_directory = "datamart/silver/clickstream/"
silver_attributes_directory  = "datamart/silver/attributes/"
silver_financials_directory  = "datamart/silver/financials/"
silver_loan_daily_directory  = "datamart/silver/loan_daily/"

for d in [silver_clickstream_directory,
          silver_attributes_directory,
          silver_financials_directory,
          silver_loan_daily_directory]:
    os.makedirs(d, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_clickstream_silver_table(
        date_str,
        bronze_lms_directory,
        silver_clickstream_directory,
        spark)

    utils.data_processing_silver_table.process_attributes_silver_table(
        date_str,
        bronze_lms_directory,
        silver_attributes_directory,
        spark)

    utils.data_processing_silver_table.process_financials_silver_table(
        date_str,
        bronze_lms_directory,
        silver_financials_directory,
        spark)

    utils.data_processing_silver_table.process_loan_daily_silver_table(
        date_str,
        bronze_lms_directory,
        silver_loan_daily_directory,
        spark)



# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"
os.makedirs(gold_feature_store_directory, exist_ok=True)

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_features_gold_table(
        date_str,
        silver_clickstream_directory,
        silver_attributes_directory,
        silver_financials_directory,
        silver_loan_daily_directory,      # passed but not yet used for join
        gold_feature_store_directory,
        spark)

# collect every snapshot folder into a list
feat_root  = "datamart/gold/feature_store"
snap_dirs  = glob.glob(os.path.join(feat_root, "gold_feature_store_*"))

print(f"Found {len(snap_dirs)} monthly snapshots")
assert snap_dirs, "No feature-store snapshots found!"

# read each folder separately
dfs = [spark.read.parquet(p) for p in snap_dirs]

# union them into one big DataFrame
full_features_df: DataFrame = reduce(
    lambda left, right: left.unionByName(right), dfs)

# quick peek at the combined data
print("Row count:", full_features_df.count())
full_features_df.printSchema()
full_features_df.show(20, truncate=False)

    
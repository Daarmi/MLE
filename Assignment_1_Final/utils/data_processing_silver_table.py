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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_clickstream_silver_table(snapshot_date_str, bronze_lms_directory, silver_clickstream_dir, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type

    # a.  fe_1 … fe_20  →  int; drop rows where cast fails
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    for c in fe_cols:
        df = df.withColumn(c, F.col(c).cast(IntegerType()))

    df = df.dropna(subset=fe_cols)          # any fe_N still NULL  ⇒ bad row

    # b.  Customer_ID regex  (keep only valid IDs)
    cust_pattern = r"^CUS_0x[0-9a-fA-F]+$"
    df = df.filter(F.col("Customer_ID").rlike(cust_pattern))

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_attributes_silver_table(
        snapshot_date_str,
        bronze_lms_directory,
        silver_attributes_dir,
        spark):
    """
    Cleans the bronze attributes snapshot for `snapshot_date_str`
    and writes a Parquet file into the silver zone.
    """

    # ── 1. read bronze ───────────────────────────────────────────────
    in_name  = f"bronze_features_attributes_{snapshot_date_str.replace('-','_')}.csv"
    in_path  = os.path.join(bronze_lms_directory, in_name)
    df = spark.read.option("header", True).csv(in_path)
    print("loaded:", in_path, "rows:", df.count())

    # ── 2. drop Name column (if present) ─────────────────────────────
    if "Name" in df.columns:
        df = df.drop("Name")

    # ── 3. Customer_ID check ─────────────────────────────────────────
    cust_re = r"^CUS_0x[0-9a-fA-F]+$"
    df = df.filter(F.col("Customer_ID").rlike(cust_re))

    # ── 4. SSN validity ──────────────────────────────────────────────
    ssn_re = r"^\d{3}-\d{2}-\d{4}$"
    if "SSN" in df.columns:               # skip if SSN isn’t in this file
        df = df.filter(F.col("SSN").rlike(ssn_re))

    # ── 5. Age cleaning & bucketing ──────────────────────────────────
    if "Age" in df.columns:
        df = (
            df.withColumn("Age", F.regexp_replace(F.col("Age").cast("string"), "_", ""))
              .withColumn("Age", F.col("Age").cast(IntegerType()))
              .dropna(subset=["Age"])
        )

        df = df.withColumn(
            "Age_Group",
            F.when(F.col("Age") < 18, "<18")
             .when((F.col("Age") <= 25), "18-25")
             .when((F.col("Age") <= 35), "26-35")
             .when((F.col("Age") <= 45), "36-45")
             .when((F.col("Age") <= 55), "46-55")
             .when((F.col("Age") <= 65), "56-65")
             .when((F.col("Age") <= 75), "66-75")
             .otherwise("76+")
        )

        # remove buckets we don’t want
        df = df.filter(~F.col("Age_Group").isin("<18", "66-75", "76+"))

    # ── 6. minimal type casting for key columns ──────────────────────
    cast_map = {
        "Customer_ID"  : StringType(),
        "SSN"          : StringType(),
        "snapshot_date": DateType(),   # if present
        "Age"          : IntegerType(),
        "Age_Group"    : StringType(),
    }
    for c, dt in cast_map.items():
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(dt))

    # ── 7. write silver ──────────────────────────────────────────────
    out_name = f"silver_features_attributes_{snapshot_date_str.replace('-','_')}"
    out_path = os.path.join(silver_attributes_dir, out_name)

    (df.write
       .mode("overwrite")
       .partitionBy("snapshot_date")       # keeps things tidy if column exists
       .parquet(out_path))

    print("saved to:", out_path, "rows kept:", df.count())
    return df


def process_financials_silver_table(
        snapshot_date_str,
        bronze_lms_directory,          # directory holding the bronze financial CSVs
        silver_financials_dir,   # directory to write the silver output
        spark):
    """
    Cleans the bronze financials snapshot for `snapshot_date_str`
    and writes a typed Parquet file to the silver zone.
    """

    # ── 1.  read bronze ──────────────────────────────────────────────
    in_part = f"bronze_features_financials_{snapshot_date_str.replace('-','_')}.csv"
    in_path = os.path.join(bronze_lms_directory, in_part)
    df = spark.read.option("header", True).csv(in_path)
    print("loaded:", in_path, "rows:", df.count())

    # ── 2.  Customer_ID validity ─────────────────────────────────────
    cust_re = r"^CUS_0x[0-9a-fA-F]+$"
    df = df.filter(F.col("Customer_ID").rlike(cust_re))

    # ── 3.  integer columns  (NaN → 0, keep ≥ 0) ─────────────────────
    int_cols = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries"
    ]
    for c in int_cols:
        if c in df.columns:
            df = (df.withColumn(c, F.col(c).cast("double"))
                    .withColumn(c, F.when(F.col(c).isNull(), 0).otherwise(F.col(c)))
                    .withColumn(c, F.col(c).cast(IntegerType())))

    # ── 4.  Annual_Income  (strip junk, to 2 dp) ─────────────────────
    if "Annual_Income" in df.columns:
        df = (
            df.withColumn("Annual_Income",
                          F.regexp_replace("Annual_Income", r"[^\d.]", ""))
              .withColumn("Annual_Income",
                          F.round(F.col("Annual_Income").cast(DoubleType()), 2))
        )

    # ── 5.  Monthly_Inhand_Salary  (numeric, 2 dp) ───────────────────
    if "Monthly_Inhand_Salary" in df.columns:
        df = (
            df.withColumn("Monthly_Inhand_Salary",
                          F.regexp_replace("Monthly_Inhand_Salary", r"[^\d.]", ""))
              .withColumn("Monthly_Inhand_Salary",
                          F.round(F.col("Monthly_Inhand_Salary").cast(DoubleType()), 2))
        )

    # ── 6.  Num_of_Loan  ↔ Type_of_Loan consistency ──────────────────
    if {"Num_of_Loan", "Type_of_Loan"}.issubset(df.columns):
        df = df.filter(~(
            ((F.col("Num_of_Loan") > 0) & (F.col("Type_of_Loan").isNull() | (F.trim("Type_of_Loan") == ""))) |
            ((F.col("Num_of_Loan") == 0) & (F.col("Type_of_Loan").isNotNull()) & (F.trim("Type_of_Loan") != ""))
        )).dropna(subset=["Type_of_Loan"])  # remove rows where Type_of_Loan is NaN

    # ── 7.  Changed_Credit_Limit  (numeric ≥ 0, 2 dp) ────────────────
    if "Changed_Credit_Limit" in df.columns:
        df = (
            df.withColumn("Changed_Credit_Limit",
                          F.regexp_replace("Changed_Credit_Limit", r"[^\d.]", ""))
              .withColumn("Changed_Credit_Limit",
                          F.round(F.col("Changed_Credit_Limit").cast(DoubleType()), 2))
              .filter(F.col("Changed_Credit_Limit") >= 0)
        )

    # ── 8.  Outstanding_Debt  (remove junk, NaN→0, 2 dp) ─────────────
    if "Outstanding_Debt" in df.columns:
        df = (
            df.withColumn("Outstanding_Debt",
                          F.regexp_replace("Outstanding_Debt", r"[^\d.]", ""))
              .withColumn("Outstanding_Debt",
                          F.round(F.col("Outstanding_Debt").cast(DoubleType()), 2))
              .fillna({"Outstanding_Debt": 0})
        )

    # ── 9.  Credit_Utilization_Ratio  (numeric, ≥0, 4 dp) ────────────
    if "Credit_Utilization_Ratio" in df.columns:
        df = (
            df.withColumn("Credit_Utilization_Ratio",
                          F.regexp_replace("Credit_Utilization_Ratio", r"[^\d.]", ""))
              .withColumn("Credit_Utilization_Ratio",
                          F.round(F.col("Credit_Utilization_Ratio").cast(DoubleType()), 4))
              .fillna({"Credit_Utilization_Ratio": 0})
              .filter(F.col("Credit_Utilization_Ratio") >= 0)
        )

    # ── 10.  Payment_of_Min_Amount  (Yes / No / NM / Unknown) ────────
    if "Payment_of_Min_Amount" in df.columns:
        df = df.withColumn(
            "Payment_of_Min_Amount",
            F.when(F.upper(F.trim("Payment_of_Min_Amount")).isin("YES", "NO", "NM"),
                   F.initcap("Payment_of_Min_Amount"))  # keep proper-case
             .otherwise(F.lit("Unknown"))
        )

    # ── 11.  Total_EMI_per_month  (numeric ≥0, 4 dp) ─────────────────
    if "Total_EMI_per_month" in df.columns:
        df = (
            df.withColumn("Total_EMI_per_month",
                          F.regexp_replace("Total_EMI_per_month", r"[^\d.]", ""))
              .withColumn("Total_EMI_per_month",
                          F.round(F.col("Total_EMI_per_month").cast(DoubleType()), 4))
              .fillna({"Total_EMI_per_month": 0})
              .filter(F.col("Total_EMI_per_month") >= 0)
        )

    # ── 12.  Amount_invested_monthly  (numeric ≥0, 4 dp) ─────────────
    if "Amount_invested_monthly" in df.columns:
        df = (
            df.withColumn("Amount_invested_monthly",
                          F.regexp_replace("Amount_invested_monthly", r"[^\d.]", ""))
              .withColumn("Amount_invested_monthly",
                          F.round(F.col("Amount_invested_monthly").cast(DoubleType()), 4))
              .fillna({"Amount_invested_monthly": 0})
              .filter(F.col("Amount_invested_monthly") >= 0)
        )

    # ── 13.  Payment_Behaviour  → Spend_Level  &  Payment_Value ──────
    if "Payment_Behaviour" in df.columns:
        df = df.withColumn("Spend_Level",
                           F.regexp_extract("Payment_Behaviour",
                                            r'^(High|Low)_spent_', 1)) \
               .withColumn("Payment_Value",
                           F.regexp_extract("Payment_Behaviour",
                                            r'_spent_(Small|Medium|Large)_value$', 1))

    # ── 14.  Monthly_Balance  (numeric, allow 0, no negatives) ───────
    if "Monthly_Balance" in df.columns:
        df = (
            df.withColumn("Monthly_Balance",
                          F.regexp_replace("Monthly_Balance", r"[^\d.]", ""))
              .withColumn("Monthly_Balance",
                          F.round(F.col("Monthly_Balance").cast(DoubleType()), 4))
              .fillna({"Monthly_Balance": 0})
              .filter(F.col("Monthly_Balance") >= 0)
        )

    # ── 15.  Credit_History_Age  →  TODO  ────────────────────────────
    # Placeholder – cleaning rules still to be defined
    # df = df.withColumn("Credit_History_Age", ...)


    # ── 17.  write silver ────────────────────────────────────────────
    out_part = f"silver_features_financials_{snapshot_date_str.replace('-','_')}"
    out_path = os.path.join(silver_financials_dir, out_part)
    (df.write
       .mode("overwrite")
       .partitionBy("snapshot_date")
       .parquet(out_path))

    print("saved to:", out_path, "rows kept:", df.count())
    return df

def process_loan_daily_silver_table(
        snapshot_date_str: str,
        bronze_lms_directory: str,
        silver_loan_daily_dir: str,
        spark):
    """
    Cleans the bronze loan-daily snapshot for `snapshot_date_str`
    and writes a typed Parquet file into the silver zone.
    """

    # ── 1. read bronze ──────────────────────────────────────────────
    in_part = f"bronze_loan_daily_{snapshot_date_str.replace('-','_')}.csv"
    in_path = os.path.join(bronze_lms_directory, in_part)
    df = spark.read.option("header", True).csv(in_path)
    print("loaded:", in_path, "rows:", df.count())

    # ── 2. loan_id validity ────────────────────────────────────────
    loan_id_re = r"^CUS_0x[0-9a-fA-F]+_\d{4}_\d{2}_\d{2}$"
    df = df.filter(F.col("loan_id").rlike(loan_id_re))

    # ── 3. Customer_ID validity ────────────────────────────────────
    cust_re = r"^CUS_0x[0-9a-fA-F]+$"
    df = df.filter(F.col("Customer_ID").rlike(cust_re))

    # ── 4. loan_start_date  →  DateType (keep rows that parse) ─────
    if "loan_start_date" in df.columns:
        df = df.withColumn("loan_start_date",
                           F.to_date("loan_start_date", "yyyy-MM-dd")) \
               .dropna(subset=["loan_start_date"])

    # ── 5. tenure & installment_num  →  IntegerType (drop non-ints) ─
    int_cols = ["tenure", "installment_num"]
    for c in int_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType())) \
                   .filter(F.col(c).isNotNull() & (F.col(c) == F.floor(F.col(c)))) \
                   .withColumn(c, F.col(c).cast(IntegerType()))

    # ── 6. money / numeric columns  →  DoubleType, NaN → 0 ─────────
    money_cols = [
        "loan_amt", "due_amt", "paid_amt",
        "overdue_amt", "balance"
    ]
    for c in money_cols:
        if c in df.columns:
            df = df.withColumn(
                    c,
                    F.regexp_replace(c, r"[^\d.]", "")
                ).withColumn(
                    c,
                    F.col(c).cast(DoubleType())
                ).fillna({c: 0})

    # ── 7. minimal cast map for key fields ─────────────────────────
    cast_map = {
        "loan_id"       : StringType(),
        "Customer_ID"   : StringType(),
        "snapshot_date" : DateType()
    }
    for c, dt in cast_map.items():
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(dt))

    # ── 8. write silver ────────────────────────────────────────────
    out_part = f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet"
    out_path = os.path.join(silver_loan_daily_dir, out_part)
    (df.write
       .mode("overwrite")
       .partitionBy("snapshot_date")
       .parquet(out_path))

    print("saved to:", out_path, "rows kept:", df.count())
    return df

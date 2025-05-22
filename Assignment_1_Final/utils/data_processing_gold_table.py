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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# ------------------------------------------------------------------
# Features  →  GOLD
# ------------------------------------------------------------------
def process_features_gold_table(
        snapshot_date_str: str,
        silver_clickstream_dir: str,
        silver_attributes_dir: str,
        silver_financials_dir: str,
        silver_loan_daily_dir: str,      # keep in signature for future use
        gold_feature_store_dir: str,
        spark):
    """
    Creates a point-in-time feature table (one row per Customer_ID) by
    joining the three “unique-key” silver snapshots:

        • clickstream  – fe_1 … fe_20
        • attributes   – Age, Occupation
        • financials   – all cleaned columns

    Parameters
    ----------
    snapshot_date_str : "YYYY-MM-DD"
    silver_*_dir      : folder holding silver_<dataset>_<date>/ parquet
    gold_feature_store_dir : where to write gold_feature_store_<date>.parquet
    """

    snap_fmt = snapshot_date_str.replace('-', '_')

    # ── 1. read the three silver snapshots ──────────────────────────
    c_path = os.path.join(
        silver_clickstream_dir, f"silver_feature_clickstream_{snap_fmt}.parquet")
    a_path = os.path.join(
        silver_attributes_dir, f"silver_features_attributes_{snap_fmt}")
    f_path = os.path.join(
        silver_financials_dir, f"silver_features_financials_{snap_fmt}")

    click_df = (spark.read.parquet(c_path)
                      .select("Customer_ID", "snapshot_date",
                              *[f"fe_{i}" for i in range(1, 21)]))

    attr_df = (spark.read.parquet(a_path)
                     .select("Customer_ID", "snapshot_date",
                             "Age", "Occupation"))

    fin_df  = spark.read.parquet(f_path)   # keep **all** columns

    # ── 2. sanity – ensure uniqueness in each base df ───────────────
    # (financials + attributes are already 1-row-per-customer per spec)
    # clickstream likewise after silver cleaning
    # If duplicates show up, keep the first record
    for df_name, df in [("click_df", click_df),
                        ("attr_df",  attr_df),
                        ("fin_df",   fin_df)]:
        dup_count = (df
                     .groupBy("Customer_ID", "snapshot_date")
                     .count()
                     .filter("count > 1")
                     .count())
        if dup_count:
            print(f"[WARN] {dup_count} duplicate rows in {df_name}; "
                  "keeping first instance.")
            window = F.row_number().over(
                        Window.partitionBy("Customer_ID", "snapshot_date")
                              .orderBy(F.monotonically_increasing_id()))
            df = df.withColumn("rn", window).filter("rn = 1").drop("rn")

    # ── 3. join order: financials (base) → click → attributes ───────
    gold_df = (fin_df
               .join(click_df, ["Customer_ID", "snapshot_date"], "left")
               .join(attr_df,  ["Customer_ID", "snapshot_date"], "left"))

    # ── 4. optional loan aggregation (commented) ────────────────────
    # If each customer can have many loan rows per date you can derive
    # summary metrics then merge:
    #
    # loan_path = os.path.join(
    #     silver_loan_daily_dir, f"silver_loan_daily_{snap_fmt}.parquet")
    # loan_df  = spark.read.parquet(loan_path)
    #
    # loan_agg = (loan_df
    #             .groupBy("Customer_ID", "snapshot_date")
    #             .agg(F.count("*").alias("loan_cnt"),
    #                  F.sum("loan_amt").alias("loan_amt_sum"),
    #                  F.max("overdue_amt").alias("max_overdue")))
    #
    # gold_df = gold_df.join(loan_agg,
    #                        ["Customer_ID", "snapshot_date"], "left")

    # ── 5. write the gold feature table ─────────────────────────────
    out_part = f"gold_feature_store_{snap_fmt}"
    out_path = os.path.join(gold_feature_store_dir, out_part)

    (gold_df.write
           .mode("overwrite")
           .partitionBy("snapshot_date")
           .parquet(out_path))

    print(f"[GOLD FEATURES] {snapshot_date_str}: "
          f"{gold_df.count():,d} rows → {out_path}")

    return gold_df

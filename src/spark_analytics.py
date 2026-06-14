"""PySpark Analytics Module for E-Commerce Data Analysis.

This module provides a SalesAnalytics class for performing advanced analytics on
e-commerce data including customer revenue analysis, category trends, and growth metrics.
"""

import logging
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# Import centralized configuration utilities
from src.config import setup_logging

logger = setup_logging(__name__)


class SalesAnalytics:
    """Encapsulates optimized PySpark workflows for high-volume e-commerce analysis."""

    def __init__(self) -> None:
        """Initializes the analytics class and provisions a lazy-loaded Spark session."""
        self.spark: Optional[SparkSession] = None

    def create_spark_session(self, app_name: str = "ShopMaxAnalytics") -> SparkSession:
        """Configures and activates an optimized local-mode SparkSession.

        Applies Kryo serialization, Adaptive Query Execution (AQE), and memory 
        management configurations designed to process 10M rows smoothly.

        Args:
            app_name (str): Monitored name designation for the Spark execution context.

        Returns:
            SparkSession: Fully instantiated and optimized active Spark environment.
        """
        try:
            logger.info("Initializing optimized SparkSession context...")
            
            self.spark = (
                SparkSession.builder
                .appName(app_name)
                .master("local[*]")  # Leverage all available local hardware threads
                .config("spark.driver.memory", "4g")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.shuffle.partitions", "32")  # Eliminates local task scheduling lag
                .getOrCreate()
            )
            
            # Reduce unnecessary verbose internal logging from Spark runtime
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("SparkSession successfully created with custom hardware optimizations.")
            return self.spark
            
        except Exception as err:
            logger.error(f"Failed to provision SparkSession context: {str(err)}", exc_info=True)
            raise

    def load_parquet(self, path: str) -> DataFrame:
        """Reads target Parquet assets into a distributed Spark DataFrame.

        Args:
            path (str): Fully qualified or relative file system location string.

        Returns:
            DataFrame: Evaluated Spark representation of specified target data matrix.
        """
        if not self.spark:
            raise RuntimeError("Active SparkSession not discovered. Execute 'create_spark_session()' first.")
        
        try:
            logger.info(f"Loading raw Parquet dataset from storage path: {path}")
            return self.spark.read.parquet(path)
        except Exception as err:
            logger.error(f"Data ingestion pipeline layer failure on path: {path}. Reason: {str(err)}")
            raise

    def top_customers_by_revenue(self, orders_df: DataFrame, products_df: DataFrame, n: int = 10) -> DataFrame:
        """Identifies top consumer profiles ranked by cumulative financial expenditures.

        Args:
            orders_df (DataFrame): Fact records containing transaction logs.
            products_df (DataFrame): Dimension records containing item costs.
            n (int): Slice limit constraint for filtering execution bounds.

        Returns:
            DataFrame: Computed collection containing sorted customer identification metrics.
        """
        try:
            logger.info(f"Calculating Top {n} revenue-generating customer profiles...")
            
            # Calculate linear revenue itemizations prior to grouping constraints
            joined_df = orders_df.join(products_df, on="product_id", how="inner")
            revenue_df = joined_df.withColumn("line_item_revenue", F.col("quantity") * F.col("price"))
            
            agg_customers = (
                revenue_df.groupBy("customer_id")
                .agg(F.round(F.sum("line_item_revenue"), 2).alias("total_spend"))
                .orderBy(F.col("total_spend").desc())
                .limit(n)
            )
            return agg_customers
        except Exception as err:
            logger.error(f"Aggregation phase calculation failure on top consumers: {str(err)}")
            raise

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Generates item sales insights grouped by daily calendar periods and categorization tag.

        Args:
            orders_df (DataFrame): Transaction operational data records.
            products_df (DataFrame): Dimension descriptive product definitions.

        Returns:
            DataFrame: Aggregated tracking matrix showing unit shifts and net cash yield.
        """
        try:
            logger.info("Compiling daily sales metrics grouped by product vertical categorization...")
            
            joined_df = orders_df.join(products_df, on="product_id", how="inner")
            
            # Extract pure date representations from standard timestamp parameters
            transformed_df = joined_df.withColumn("order_day", F.to_date(F.col("order_date")))
            transformed_df = transformed_df.withColumn("line_revenue", F.col("quantity") * F.col("price"))
            
            daily_category_report = (
                transformed_df.groupBy("order_day", "category")
                .agg(
                    F.round(F.sum("line_revenue"), 2).alias("daily_revenue"),
                    F.sum("quantity").alias("units_sold")
                )
                .orderBy(F.col("order_day").desc(), F.col("daily_revenue").desc())
            )
            return daily_category_report
        except Exception as err:
            logger.error(f"Distributed computation failure during daily category sequence run: {str(err)}")
            raise

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Computes Month-over-Month (MoM) revenue growth trends utilizing analytical Window frames.

        Args:
            orders_df (DataFrame): Fact transaction metrics logs.
            products_df (DataFrame): Dimension tracking product items table.

        Returns:
            DataFrame: Context containing structured monthly trajectories and growth analytics.
        """
        try:
            logger.info("Executing analytical window evaluation for Month-over-Month growth trends...")
            
            joined_df = orders_df.join(products_df, on="product_id", how="inner")
            
            # Construct standard chronological yyyy-MM evaluation string formats
            monthly_revenue_df = (
                joined_df.withColumn("year_month", F.date_format(F.col("order_date"), "yyyy-MM"))
                .withColumn("line_revenue", F.col("quantity") * F.col("price"))
                .groupBy("year_month")
                .agg(F.round(F.sum("line_revenue"), 2).alias("monthly_revenue"))
            )
            
            # Establish window parameters ordered sequentially by year-month strings
            window_spec = Window.orderBy("year_month")
            
            # Access previous period indices using historical lookbacks (LAG)
            trends_df = monthly_revenue_df.withColumn(
                "previous_month_revenue", 
                F.lag("monthly_revenue", 1).over(window_spec)
            )
            
            # Compute MoM growth percentage rate
            # We use coalesce to handle the first month where previous_month_revenue is null
            growth_calculation = (
                (F.col("monthly_revenue") - F.col("previous_month_revenue")) / 
                F.col("previous_month_revenue")
            ) * 100
            
            final_trends_df = trends_df.withColumn(
                "mom_growth_pct",
                F.coalesce(F.round(growth_calculation, 2), F.lit(0.0))
            ).orderBy("year_month")
            
            return final_trends_df
        except Exception as err:
            logger.error(f"Analytical Window execution failure inside monthly trends routine: {str(err)}")
            raise
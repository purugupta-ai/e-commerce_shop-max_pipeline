"""Shop-max Analytics Execution Pipeline.

This script acts as the orchestration layer to load raw e-commerce Parquet files,
execute distributed PySpark analytical transformations, and output the resulting
business intelligence reports to the console in tabular format.
"""

import sys
import time
from src.config import (
    setup_logging, 
    CUSTOMERS_FILE, 
    PRODUCTS_FILE, 
    ORDERS_FILE
)
from src.spark_analytics import SalesAnalytics

# Initialize system logger
logger = setup_logging(__name__)


def main() -> None:
    """Orchestrates data loading, transformation, and tabular reporting.
    
    Ensures safe initialization and teardown of JVM/Spark resources using 
    strict try/finally blocks to prevent memory leaks on the host machine.
    """
    logger.info("==================================================")
    logger.info("📊 Starting Shop-max PySpark Analytics Engine 📊")
    logger.info("==================================================")

    start_time = time.perf_counter()
    analytics = SalesAnalytics()
    
    try:
        # 1. Initialize JVM / Spark Session
        spark = analytics.create_spark_session(app_name="ShopMax_Reporting")

        # 2. Ingest Parquet Data
        logger.info("Loading Parquet datasets into Spark DataFrames...")
        customers_df = analytics.load_parquet(str(CUSTOMERS_FILE))
        products_df = analytics.load_parquet(str(PRODUCTS_FILE))
        orders_df = analytics.load_parquet(str(ORDERS_FILE))

        # 3. Generate and Display Reports
        logger.info("\n" + "="*50)
        logger.info("REPORT 1: TOP 10 CUSTOMERS BY REVENUE")
        logger.info("="*50)
        top_customers = analytics.top_customers_by_revenue(orders_df, products_df, n=10)
        top_customers.show(truncate=False)

        logger.info("\n" + "="*50)
        logger.info("REPORT 2: DAILY SALES BY CATEGORY (Top 20)")
        logger.info("="*50)
        category_sales = analytics.sales_by_category(orders_df, products_df)
        category_sales.show(20, truncate=False)

        logger.info("\n" + "="*50)
        logger.info("REPORT 3: MONTH-OVER-MONTH REVENUE TRENDS")
        logger.info("="*50)
        monthly_trends = analytics.monthly_trends(orders_df, products_df)
        monthly_trends.show(truncate=False)

        # 4. Telemetry
        execution_time = time.perf_counter() - start_time
        logger.info(f"✅ Analytics processing completed successfully in {execution_time:.2f} seconds.")

    except Exception as err:
        logger.critical("Fatal error encountered during PySpark execution.", exc_info=True)
        sys.exit(1)
        
    finally:
        # CRITICAL: Always release Spark JVM memory regardless of success or failure
        if analytics.spark:
            logger.info("Stopping Spark session and releasing local memory...")
            analytics.spark.stop()
            logger.info("Spark session cleanly terminated.")


if __name__ == "__main__":
    main()
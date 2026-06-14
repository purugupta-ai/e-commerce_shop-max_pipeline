"""Shop-max E-commerce Pipeline Execution Entrypoint.

This script orchestrates the generation of synthetic e-commerce datasets
(customers, products, and orders) and ensures structured persistence into 
the raw landing zone.
"""

import sys
import time
from src.config import setup_logging
from src.data_generator import SyntheticDataGenerator

# Initialize structured logger for system monitoring
logger = setup_logging(__name__)


def main() -> None:
    """Orchestrates the execution lifecycle of the data generation pipeline.

    Instantiates the underlying generation framework, tracks high-precision 
    telemetry metrics, and guarantees explicit error capture along with 
    appropriate system termination signals.

    Returns:
        None
    """
    logger.info("==========================================================")
    logger.info("Starting 'Shop-max' Synthetic Data Generation Execution...")
    logger.info("==========================================================")

    start_time = time.perf_counter()

    try:
        # Initialize the high-performance synthetic data generator
        logger.info("Initializing SyntheticDataGenerator instance...")
        generator = SyntheticDataGenerator()

        # Run the vectorized data generation and parquet file storage pipeline
        logger.info("Executing generation sequence (Customers -> Products -> Orders)...")
        generator.run_pipeline()

        # Calculate operational runtime metrics
        end_time = time.perf_counter()
        execution_duration = end_time - start_time

        logger.info("==========================================================")
        logger.info("🎉 'Shop-max' Pipeline Execution Completed Successfully!")
        logger.info(f"⏱️ Total Wall-Clock Runtime: {execution_duration:.2f} seconds")
        logger.info("==========================================================")

    except KeyboardInterrupt:
        logger.warning("Pipeline execution forcefully aborted by user signal (SIGINT).")
        sys.exit(130)

    except Exception as err:
        logger.critical(
            "Fatal pipeline exception encountered during execution lifecycle.",
            exc_info=True
        )
        logger.error(f"Failure Reason: {str(err)}")
        logger.info("==========================================================")
        
        # Propagate a non-zero exit status code to inform orchestrators (Airflow, Prefect, Cron, etc.)
        sys.exit(1)


if __name__ == "__main__":
    main()
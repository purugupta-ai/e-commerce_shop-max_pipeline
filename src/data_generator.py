import time
from typing import Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker
from tqdm import tqdm

# Import centralized configurations and logger
from src.config import (
    setup_logging, 
    CUSTOMERS_FILE, 
    PRODUCTS_FILE, 
    ORDERS_FILE
)

logger = setup_logging(__name__)

class SyntheticDataGenerator:
    """
    A high-performance generator for synthetic e-commerce data using 
    NumPy vectorization and PyArrow chunking to prevent OOM errors.
    """

    def __init__(
        self, 
        num_customers: int = 500_000, 
        num_products: int = 50_000, 
        num_orders: int = 10_000_000, 
        random_seed: int = 42
    ) -> None:
        """
        Initializes the generator with scale targets and random seeds.

        Args:
            num_customers (int): Number of customers to generate.
            num_products (int): Number of products to generate.
            num_orders (int): Number of orders to generate.
            random_seed (int): Random seed for reproducibility.
        """
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        
        # Set seeds
        np.random.seed(random_seed)
        self.fake = Faker()
        Faker.seed(random_seed)
        
        # Date boundaries for the dataset
        self.start_date = pd.Timestamp('2021-01-01')
        self.end_date = pd.Timestamp('2026-01-01')

    def generate_customers(self) -> pd.DataFrame:
        """
        Generates the customers dimension table.

        Returns:
            pd.DataFrame: A DataFrame containing generated customer data.
        """
        logger.info(f"Generating {self.num_customers} customers...")
        
        customer_ids = np.arange(1, self.num_customers + 1)
        ages = np.clip(np.random.normal(loc=35, scale=10, size=self.num_customers), 18, 90).astype(int)
        
        names, emails, cities, countries, reg_dates = [], [], [], [], []
        
        for _ in tqdm(range(self.num_customers), desc="Faker Customers"):
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            names.append(f"{first_name} {last_name}")
            emails.append(f"{first_name.lower()}.{last_name.lower()}@{self.fake.free_email_domain()}")
            cities.append(self.fake.city())
            countries.append(self.fake.country())
            reg_dates.append(self.fake.date_between_dates(date_start=self.start_date, date_end=self.end_date))

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "name": names,
            "email": emails,
            "age": ages,
            "city": cities,
            "country": countries,
            "registration_date": pd.to_datetime(reg_dates).astype('datetime64[us]')
        })
        
        df.to_parquet(CUSTOMERS_FILE, engine='pyarrow', index=False)
        logger.info(f"Saved customers to {CUSTOMERS_FILE}")
        return df

    def generate_products(self) -> pd.DataFrame:
        """
        Generates the products dimension table.

        Returns:
            pd.DataFrame: A DataFrame containing generated product data.
        """
        logger.info(f"Generating {self.num_products} products...")
        
        categories = [
            "Electronics", "Apparel", "Home & Garden", "Sports", 
            "Health & Beauty", "Toys", "Automotive", "Grocery"
        ]
        
        product_ids = np.arange(1, self.num_products + 1)
        prices = np.round(np.random.uniform(5.0, 1200.0, size=self.num_products), 2)
        stocks = np.random.randint(0, 2001, size=self.num_products)
        ratings = np.round(np.random.uniform(1.0, 5.0, size=self.num_products), 1)
        assigned_categories = np.random.choice(categories, size=self.num_products)
        
        names = []
        for i in tqdm(range(self.num_products), desc="Faker Products"):
            cat_suffix = assigned_categories[i].split()[0]
            names.append(f"{self.fake.word().capitalize()} {cat_suffix}")

        df = pd.DataFrame({
            "product_id": product_ids,
            "product_name": names,
            "category": assigned_categories,
            "price": prices,
            "stock": stocks,
            "rating": ratings
        })
        
        df.to_parquet(PRODUCTS_FILE, engine='pyarrow', index=False)
        logger.info(f"Saved products to {PRODUCTS_FILE}")
        return df

    def generate_orders(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the orders fact table in chunks directly to Parquet.
        Applies Pareto distribution (80/20) and ensures logical temporal sequence.

        Returns:
            pd.DataFrame: The complete generated orders DataFrame.
        Args:
            customers_df (pd.DataFrame): The generated customers DataFrame to reference registration dates.
        """
        logger.info(f"Generating {self.num_orders} orders in chunks using Pareto distributions...")
        
        # Calculate Pareto probabilities for the 80/20 rule
        # A Pareto shape parameter (alpha) of ~1.16 approximates the 80/20 rule
        cust_weights = np.random.pareto(a=1.16, size=self.num_customers) + 1.0
        cust_probs = cust_weights / cust_weights.sum()
        
        prod_weights = np.random.pareto(a=1.16, size=self.num_products) + 1.0
        prod_probs = prod_weights / prod_weights.sum()

        chunk_size = 1_000_000
        num_chunks = self.num_orders // chunk_size
        
        # Pre-extract values for fast NumPy indexing
        customer_id_array = customers_df["customer_id"].values
        product_id_array = np.arange(1, self.num_products + 1)
        
        # Convert dates to seconds for fast vectorized random date calculation
        reg_dates_sec = customers_df["registration_date"].astype('int64').values // 10**9
        max_date_sec = int(self.end_date.timestamp())

        # Define schema for the pyarrow Parquet writer
        schema = pa.schema([
            ("order_id", pa.int64()),
            ("customer_id", pa.int64()),
            ("product_id", pa.int64()),
            ("quantity", pa.int32()),
            ("order_date", pa.timestamp('us'))
        ])

        all_chunks = []
        with pq.ParquetWriter(ORDERS_FILE, schema) as writer:
            for chunk_idx in tqdm(range(num_chunks), desc="Generating Orders (Chunks)"):
                start_id = (chunk_idx * chunk_size) + 1
                order_ids = np.arange(start_id, start_id + chunk_size)
                
                # Sample IDs using Pareto probabilities
                c_ids = np.random.choice(customer_id_array, size=chunk_size, p=cust_probs)
                p_ids = np.random.choice(product_id_array, size=chunk_size, p=prod_probs)
                quantities = np.random.randint(1, 6, size=chunk_size)
                
                # Vectorized temporal logic: fetch registration dates for sampled customers
                # Since customer_ids start at 1, subtract 1 to get the correct 0-based array index
                sampled_reg_sec = reg_dates_sec[c_ids - 1]
                
                # Generate random timestamps strictly between registration date and max_date
                time_deltas = np.random.randint(0, np.maximum(1, max_date_sec - sampled_reg_sec))
                order_dates_sec = sampled_reg_sec + time_deltas
                order_dates = pd.to_datetime(order_dates_sec, unit='s')
                
                chunk_df = pd.DataFrame({
                    "order_id": order_ids,
                    "customer_id": c_ids,
                    "product_id": p_ids,
                    "quantity": quantities,
                    "order_date": order_dates
                })
                
                table = pa.Table.from_pandas(chunk_df, schema=schema)
                writer.write_table(table)
                all_chunks.append(chunk_df)

        logger.info(f"Saved chunked orders to {ORDERS_FILE}")
        return pd.concat(all_chunks, ignore_index=True)

    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates all datasets and returns them as DataFrames."""
        logger.info("Starting generation of all datasets...")
        customers_df = self.generate_customers()
        products_df = self.generate_products()
        orders_df = self.generate_orders(customers_df)
        return customers_df, products_df, orders_df

    def run_pipeline(self) -> None:
        """Executes the full generation pipeline in sequence."""
        start_time = time.time()
        logger.info("Initializing Data Generation Pipeline...")
        
        self.generate_all()
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    try:
        generator = SyntheticDataGenerator()
        generator.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed due to an error: {str(e)}", exc_info=True)
        raise
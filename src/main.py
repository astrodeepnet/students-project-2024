import time
import pandas as pd
from database.creation import DatabaseTables
from pipeline.pipeline import DataPipeline

from database.operations import DatabaseConnector
from database.connection import DATABASE_URL


if __name__ == "__main__":
    start = time.time()
    # Load dataset
    df = pd.read_csv('../data/allStarLite1000rows.csv')
    print(f"Dataset loaded, time taken: {time.time() - start:.2f} seconds")


    connector = DatabaseConnector(DATABASE_URL)

    database_tables = DatabaseTables()

    # Drop all tables
    connector.drop_tables(DatabaseTables.Base)

    # Create all tables
    connector.create_tables(DatabaseTables.Base)

    # Initialize pipeline
    pipeline = DataPipeline(df, connector)

    # Run pipeline
    pipeline.run()

    print("################## SUCCESS ##################" )
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

import time
import pandas as pd
from database.creation import DatabaseTables
from pipeline.pipeline import DataPipeline

from database.operations import DatabaseConnector
from database.connection import DATABASE_URL

import numpy as np
import pandas as pd
from astropy.io import fits

import os

if __name__ == "__main__":
    start = time.time()

    df_path = '../data/allStarLiteFULL.csv'

    if not os.path.exists(df_path):
        print("File not found, creating it")
        hdulist = fits.open('data/allStarLite-dr17-synspec_rev1.fits')
        hdu1_data = hdulist[1].data
        df = pd.DataFrame(hdu1_data.tolist(), columns=hdu1_data.names)
    else :
        print("File found, loading it")
        df = pd.read_csv(df_path)

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

    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

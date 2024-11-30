import time
import numpy as np
import pandas as pd
from math import ceil

from tqdm import tqdm

from const import table_column_mapping, to_remove_from_main_dataset, to_remove_from_chemicals
from src.database.creation import DatabaseTables


class DataPipeline:
    def __init__(self, df: pd.DataFrame, connector):
        """
        Initialize the data pipeline with the main dataset.
        """
        self.df = df
        self.connector = connector

        # needed subsets
        self.chemical_subset = None
        self.chemical_subset_err = None
        self.chemical_subset_flag = None
        self.chemical_elements = None
        self.decile_intervals = None
        self.decile_intervals_chemical = None

        self.surveys = None
        self.telescopes = None
        self.survey_name_to_obj = {}
        self.telescope_name_to_obj = {}

        self.star_objects = []
        self.range_objects = []

    @staticmethod
    def replace_na(value):
        """
        Replace pd.NA and np.nan with None, and convert numpy data types to Python native types.
        """
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        else:
            return value

    def preprocess(self):
        """
        Preprocess the dataset: drop rows, filter columns, etc.
        """
        # Drop first row
        self.df = self.df.drop(0)

        df = self.df

        # drop duplicates APOGEE_ID
        df = df.drop_duplicates(subset='APOGEE_ID')
        print(df.shape)

        df = df.copy()

        if 'SURVEY' in df.columns:
            # Split the 'SURVEY' column into lists
            df['SURVEY_LIST'] = df['SURVEY'].apply(lambda x: str(x).split(',') if pd.notnull(x) else [])
            # Get all unique surveys
            all_surveys = set()
            df['SURVEY_LIST'].apply(lambda x: all_surveys.update(x))
            self.surveys = pd.DataFrame({
                'name': list(all_surveys)
            }).dropna().reset_index(drop=True)

        if 'SURVEY' in df.columns:
            # Split the 'SURVEY' column into lists
            df.loc[:, 'SURVEY_LIST'] = df['SURVEY'].apply(lambda x: str(x).split(',') if pd.notnull(x) else [])
            # Get all unique surveys
            all_surveys = set()
            df['SURVEY_LIST'].apply(lambda x: all_surveys.update(x))
            self.surveys = pd.DataFrame({
                'name': list(all_surveys)
            }).dropna().reset_index(drop=True)

        if 'TELESCOPE' in df.columns:
            # Split the 'TELESCOPE' column into lists
            df['TELESCOPE_LIST'] = df['TELESCOPE'].apply(lambda x: str(x).split(',') if pd.notnull(x) else [])
            # Get all unique telescopes
            all_telescopes = set()
            df['TELESCOPE_LIST'].apply(lambda x: all_telescopes.update(x))
            self.telescopes = pd.DataFrame({
                'name': list(all_telescopes)
            }).dropna().reset_index(drop=True)

        chemical_subset = df.filter(regex='_FE', axis=1)
        chemical_subset = pd.concat([chemical_subset, df.filter(regex='_H', axis=1)], axis=1)
        chemical_subset = chemical_subset.drop(to_remove_from_chemicals, axis=1) # drop a specific list of columns

        # create a df without the chemical abundances
        df = df.drop(chemical_subset.columns, axis=1)

        df = df.drop(to_remove_from_main_dataset, axis=1)

        chemical_subset = chemical_subset[chemical_subset.columns.drop(list(chemical_subset.filter(regex='_SPEC')))]

        # Add back APOGEE_ID to the chemical subset
        chemical_subset['APOGEE_ID'] = df['APOGEE_ID']

        self.df = df
        self.chemical_subset = chemical_subset

    def filter_and_clean(self):
        """
        Perform additional filtering and cleaning on chemical subsets.
        """
        # Drop columns with 100% NaN values
        self.chemical_subset = self.chemical_subset.dropna(axis=1, how='all')

        # Separate error and flag columns
        self.chemical_subset_err = self.chemical_subset.filter(regex='_ERR', axis=1)
        self.chemical_subset_flag = self.chemical_subset.filter(regex='_FLAG', axis=1)

        # Remove error and flag columns from the main chemical_subset
        self.chemical_subset = self.chemical_subset.drop(self.chemical_subset_err.columns, axis=1)
        self.chemical_subset = self.chemical_subset.drop(self.chemical_subset_flag.columns, axis=1)

        # Parse chemical elements
        self.chemical_elements = [x.split('_')[0] for x in self.chemical_subset.columns]

    def calculate_decile_intervals_all_columns(self, df, decimals=3):
        intervals = {}
        for column in tqdm(df.select_dtypes(include=[np.number]).columns, desc="Calculating decile intervals"):

            # Exclude columns with all NaN values
            valid_data = df[column].dropna()
            if not valid_data.empty:
                deciles = np.percentile(valid_data, q=np.arange(0, 101, 10))
                deciles[0] = np.floor(
                    deciles[0] * 10 ** decimals) / 10 ** decimals  # Adjust the first decile to be inclusive
                deciles[-1] = np.ceil(
                    deciles[-1] * 10 ** decimals) / 10 ** decimals  # Upper bound

                # Create intervals with rounded values
                intervals[f"{column}_RANGE"] = [
                    (round(deciles[i], decimals), round(deciles[i + 1], decimals))
                    for i in range(len(deciles) - 1)
                ]
            else:
                intervals[f"{column}_RANGE"] = [(np.nan, np.nan) for _ in range(10)]
        return pd.DataFrame.from_dict(intervals, orient='index').transpose()

    def generate_ranges(self):
        """
        Generate decile intervals for all columns in the main dataset.
        """
        self.decile_intervals = self.calculate_decile_intervals_all_columns(self.df)
        self.decile_intervals_chemical = self.calculate_decile_intervals_all_columns(self.chemical_subset)

    def prepare_for_insertion(self):
        """
        Prepare the data for database insertion.

        This method performs final data cleaning and formatting steps required
        before the data can be transformed into ORM objects and inserted into the database.
        """
        # Fill NaN values if necessary
        self.df = self.df.fillna(value=np.nan)
        self.chemical_subset = self.chemical_subset.fillna(value=np.nan)

        # Ensure correct data types
        self.df = self.df.convert_dtypes()
        self.chemical_subset = self.chemical_subset.convert_dtypes()

        # Replace pd.NA with None
        self.df = self.df.where(self.df.notna(), None)
        self.chemical_subset = self.chemical_subset.where(self.chemical_subset.notna(), None)

    def create_survey_objects(self):
        """
        Create ORM objects for surveys and insert them into the database.
        """
        if self.surveys is not None:
            survey_objects = []
            for idx, row in self.surveys.iterrows():
                survey = DatabaseTables.Survey(name=row['name'])
                survey_objects.append(survey)
            try:
                with self.connector.session_scope() as session:
                    session.bulk_save_objects(survey_objects)
                    session.commit()
                    # Build mapping from survey name to ORM object
                    surveys_in_db = session.query(DatabaseTables.Survey).all()
                    self.survey_name_to_obj = {survey.name: survey for survey in surveys_in_db}
            except Exception as e:
                print(f"An error occurred during survey insertion: {e}")

    def create_telescope_objects(self):
        """
        Create ORM objects for telescopes and insert them into the database.
        """
        if self.telescopes is not None:
            telescope_objects = []
            for idx, row in self.telescopes.iterrows():
                telescope = DatabaseTables.Telescope(name=row['name'])
                telescope_objects.append(telescope)
            try:
                with self.connector.session_scope() as session:
                    session.bulk_save_objects(telescope_objects)
                    session.commit()
                    # Build mapping from telescope name to ORM object
                    telescopes_in_db = session.query(DatabaseTables.Telescope).all()
                    self.telescope_name_to_obj = {telescope.name: telescope for telescope in telescopes_in_db}
            except Exception as e:
                print(f"An error occurred during telescope insertion: {e}")
    def create_orm_objects(self, batch_size=1000):
        """
        Create ORM objects from the processed DataFrame and insert them into the database in batches.
        """

        total_rows = self.df.shape[0]
        num_batches = ceil(total_rows / batch_size)
        count = 0

        for batch_num in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_rows)
            batch_df = self.df.iloc[start_idx:end_idx]

            star_objects = []

            for row in batch_df.itertuples(index=False, name='Row'):
                star = DatabaseTables.Star(
                    apogee_id=row.APOGEE_ID,
                    name=row.APOGEE_ID,  # Assuming name is the same as APOGEE_ID
                    nb_visits=self.replace_na(getattr(row, 'NVISITS', None)),
                    ak_wise=self.replace_na(getattr(row, 'AK_WISE', None)),
                    snr=self.replace_na(getattr(row, 'SNR', None)),
                    m_h=self.replace_na(getattr(row, 'M_H', None)),
                    m_h_err=self.replace_na(getattr(row, 'M_H_ERR', None)),
                    vsini=self.replace_na(getattr(row, 'VSINI', None)),
                    vmicro=self.replace_na(getattr(row, 'VMICRO', None)),
                    vmacro=self.replace_na(getattr(row, 'VMACRO', None)),
                    teff=self.replace_na(getattr(row, 'TEFF', None)),
                    teff_err=self.replace_na(getattr(row, 'TEFF_ERR', None)),
                    logg=self.replace_na(getattr(row, 'LOGG', None)),
                    logg_err=self.replace_na(getattr(row, 'LOGG_ERR', None)),
                    j=self.replace_na(getattr(row, 'J', None)),
                    j_err=self.replace_na(getattr(row, 'J_ERR', None)),
                    h=self.replace_na(getattr(row, 'H', None)),
                    h_err=self.replace_na(getattr(row, 'H_ERR', None)),
                    k=self.replace_na(getattr(row, 'K', None)),
                    k_err=self.replace_na(getattr(row, 'K_ERR', None)),
                )

                star.m_h_id = self.get_range_id('metalicity', star.m_h)
                star.teff_id = self.get_range_id('temperature', star.teff)
                star.logg_id = self.get_range_id('surface_gravity', star.logg)
                star.vsini_id = self.get_range_id('vsini', star.vsini)
                star.vmicro_id = self.get_range_id('vmicro', star.vmicro)
                star.vmacro_id = self.get_range_id('vmacro', star.vmacro)
                star.j_err_id = self.get_range_id('j_error', star.j_err)
                star.h_err_id = self.get_range_id('h_error', star.h_err)
                star.k_err_id = self.get_range_id('k_error', star.k_err)

                # display telescope and survey list for debugging
                print(row.APOGEE_ID)
                print(row.TELESCOPE_LIST)
                print(row.SURVEY_LIST)

                # Associate surveys
                survey_list = getattr(row, 'SURVEY_LIST', [])
                for survey_name in survey_list:
                    survey_obj = self.survey_name_to_obj.get(survey_name.strip())
                    if survey_obj:
                        star.surveys.append(survey_obj)

                # Associate telescopes
                telescope_list = getattr(row, 'TELESCOPE_LIST', [])
                for telescope_name in telescope_list:
                    telescope_obj = self.telescope_name_to_obj.get(telescope_name.strip())
                    if telescope_obj:
                        star.telescopes.append(telescope_obj)

                star_objects.append(star)
                count += 1

            # Insert batch into database
            self.insert_batch_into_database(star_objects)

        print("Number of stars:", count)

    def insert_batch_into_database(self, star_objects):
        """
        Insert a batch of ORM objects into the database.
        """
        try:
            with self.connector.session_scope() as session:
                session.add_all(star_objects)
                session.commit()
        except Exception as e:
            print(f"An error occurred during database insertion: {e}")

    def create_range_objects(self):
        """
        Create ORM objects for the range tables based on decile intervals.
        """
        # Create ranges for 'metalicity', 'temperature', etc.
        range_mappings = {
            'metalicity': ('M_H', DatabaseTables.Metalicity, 'm_h_range'),
            'temperature': ('TEFF', DatabaseTables.Temperature, 'teff_range'),
            'surface_gravity': ('LOGG', DatabaseTables.Surface_Gravity, 'logg_range'),
            'vsini': ('VSINI', DatabaseTables.Vsini, 'vsini_range'),
            'vmicro': ('VMICRO', DatabaseTables.Vmicro, 'vmicro_range'),
            'vmacro': ('VMACRO', DatabaseTables.Vmacro, 'vmacro_range'),
            'j_error': ('J_ERR', DatabaseTables.J_error, 'j_err_range'),
            'h_error': ('H_ERR', DatabaseTables.H_error, 'h_err_range'),
            'k_error': ('K_ERR', DatabaseTables.K_error, 'k_err_range'),
        }

        for table_name, (column_name, model_class, range_attr) in range_mappings.items():
            if column_name in self.df.columns:
                intervals = self.calculate_decile_intervals(self.df[column_name])
                for idx, (lower, upper) in enumerate(intervals):
                    range_id = f"{table_name.upper()}_{idx}"
                    range_instance = model_class(id=range_id)
                    setattr(range_instance, range_attr, f"{lower} to {upper}")
                    self.range_objects.append(range_instance)

        self.insert_range_objects()

    def insert_range_objects(self):
        """
        Insert range ORM objects into the database.
        """
        try:
            with self.connector.session_scope() as session:
                for obj in tqdm(self.range_objects, desc="Inserting range objects"):
                    session.merge(obj)
                print("Range objects inserted successfully!")
        except Exception as e:
            print(f"An error occurred during database insertion: {e}")

    def inser_telecope_survey(self):
        """
        Insert telescope and survey objects into the database.
        """
        self.create_survey_objects()
        self.create_telescope_objects()

        try:
            with self.connector.session_scope() as session:
                for survey in self.survey_name_to_obj.values():
                    session.merge(survey)
                for telescope in self.telescope_name_to_obj.values():
                    session.merge(telescope)
                print("Survey and telescope objects inserted successfully!")
        except Exception as e:
            print(f"An error occurred during database insertion: {e}")

    def calculate_decile_intervals(self, series, decimals=3):
        """
        Calculate decile intervals for a given pandas Series.
        """
        valid_data = series.dropna()
        if not valid_data.empty:
            deciles = np.percentile(valid_data, q=np.arange(0, 101, 10))
            deciles[0] = np.floor(deciles[0] * 10 ** decimals) / 10 ** decimals
            deciles[-1] = np.ceil(deciles[-1] * 10 ** decimals) / 10 ** decimals
            intervals = [
                (round(deciles[i], decimals), round(deciles[i + 1], decimals))
                for i in range(len(deciles) - 1)
            ]
            return intervals
        else:
            return [(np.nan, np.nan) for _ in range(10)]

    def get_range_id(self, table_name, value):
        """
        Get the range ID for a given value based on decile intervals.

        Parameters:
            table_name (str): The name of the range table.
            value (float): The value to find the range for.

        Returns:
            str: The ID of the range.
        """
        if value is None or pd.isna(value):
            return None

        # Use the mapping to get the corresponding column name
        column_name = table_column_mapping.get(table_name)
        if not column_name:
            return None

        range_col = f"{column_name}_RANGE"

        if range_col not in self.decile_intervals.columns:
            return None

        intervals_series = self.decile_intervals[range_col]

        # Iterate over the intervals
        for idx, interval in intervals_series.items():
            if pd.isna(interval):
                continue
            lower, upper = interval
            if lower <= value <= upper:
                return f"{table_name.upper()}_{idx}"
        return None

    def run(self):
        """
        Run the entire pipeline.

        This method orchestrates the data processing steps: preprocessing, cleaning,
        generating ranges, preparing for insertion, creating ORM objects, and inserting
        the data into the database.
        """
        self.preprocess()
        print("Preprocessing done")
        self.filter_and_clean()
        print("Filtering and cleaning done")
        self.generate_ranges()
        print("Ranges generated")
        self.create_range_objects()
        print("Range objects created")
        self.prepare_for_insertion()
        print("Preparation done")
        self.inser_telecope_survey()
        self.create_orm_objects()
        print("ORM objects created")

        #self.insert_into_database()
        print("Insertion done")


# Example Usage
if __name__ == "__main__":
    start = time.time()
    # Load dataset
    df = pd.read_csv('../../data/allStarLite1000rows.csv')
    print(f"Dataset loaded, time taken: {time.time() - start:.2f} seconds")

    from src.database.operations import DatabaseConnector
    from src.database.connection import DATABASE_URL

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

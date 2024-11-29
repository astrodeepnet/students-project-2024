import numpy as np
import pandas as pd
from const import table_column_mapping
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

        self.star_objects = []
        self.range_objects = []

    def replace_na(self,value):
        """
        Replace pd.NA and np.nan with None.
        """
        if pd.isna(value):
            return None
        else:
            return value

    def preprocess(self):
        """
        Preprocess the dataset: drop rows, filter columns, etc.
        """
        # Drop first row
        self.df = self.df.drop(0)

        df = self.df

        # subset all chemical abundances (collumn name contain "_FE")
        chemical_subset = df.filter(regex='_FE', axis=1)

        # merge with + df.filter(regex='_H', axis=1)
        chemical_subset = pd.concat([chemical_subset, df.filter(regex='_H', axis=1)], axis=1)

        to_remove_from_chemicals = ['RV_FEH', 'MIN_H', 'MAX_H', 'GAIAEDR3_R_HI_GEO', 'GAIAEDR3_R_HI_PHOTOGEO',
                                    'CU_FE_ERR',
                                    'P_FE_ERR', 'P_FE_FLAG',
                                    'CU_FE_FLAG',
                                    'M_H', 'M_H_ERR', 'X_H_SPEC', 'X_H', 'X_H_ERR']
        chemical_subset = chemical_subset.drop(to_remove_from_chemicals, axis=1)

        # create a df without the chemical abundances
        df = df.drop(chemical_subset.columns, axis=1)

        to_remove_from_main_dataset = ['P_FE_ERR',
                                       'P_FE_FLAG',
                                       'CU_FE_ERR', 'GAIAEDR3_PARALLAX_ERROR', 'GAIAEDR3_PMRA', 'GAIAEDR3_PMRA_ERROR',
                                       'GAIAEDR3_PMDEC', 'GAIAEDR3_PMDEC_ERROR', 'GAIAEDR3_PHOT_G_MEAN_MAG',
                                       'GAIAEDR3_PHOT_BP_MEAN_MAG', 'GAIAEDR3_PHOT_RP_MEAN_MAG',
                                       'GAIAEDR3_DR2_RADIAL_VELOCITY', 'GAIAEDR3_DR2_RADIAL_VELOCITY_ERROR',
                                       'GAIAEDR3_R_MED_GEO', 'GAIAEDR3_R_LO_GEO', 'GAIAEDR3_R_HI_GEO',
                                       'GAIAEDR3_R_MED_PHOTOGEO', 'GAIAEDR3_R_LO_PHOTOGEO',
                                       'GAIAEDR3_R_HI_PHOTOGEO', 'MEANFIB', 'SIGFIB', 'AK_TARG', 'AK_TARG_METHOD',
                                       'APOGEE_TARGET1', 'APOGEE_TARGET2', 'APOGEE2_TARGET1',
                                       'APOGEE2_TARGET2', 'APOGEE2_TARGET3', 'APOGEE2_TARGET4',
                                       'RV_CHI2', 'RV_CCFWHM', 'RV_AUTOFWHM', 'VSCATTER', 'VERR', 'RV_FEH', 'RV_FLAG',
                                       'MIN_H', 'MAX_H',
                                       'MIN_JK', 'MAX_JK', 'GAIAEDR3_SOURCE_ID', 'GAIAEDR3_PARALLAX',
                                       'ASPCAP_GRID', 'ASPCAP_CHI2', 'FRAC_BADPIX', 'FRAC_LOWSNR', 'FRAC_SIGSKY',
                                       'ELEM_CHI2', 'ELEMFRAC', 'EXTRATARG', 'MEMBERFLAG', 'MEMBER', 'X_H_SPEC',
                                       'X_M_SPEC',
                                       'FRAC_BADPIX', 'FRAC_LOWSNR', 'FRAC_SIGSKY', 'ELEM_CHI2', 'ELEMFRAC',
                                       'X_H_SPEC', 'X_M_SPEC', 'TEFF_SPEC', 'LOGG_SPEC', 'CU_FE_FLAG', 'ALT_ID',
                                       'PROGRAMNAME',
                                       'RV_TEFF', 'RV_LOGG', 'RV_ALPHA', 'RV_CARB', 'SNREV', 'SFD_EBV', 'N_COMPONENTS']

        df = df.drop(to_remove_from_main_dataset, axis=1)


        chemical_subset = chemical_subset[chemical_subset.columns.drop(list(chemical_subset.filter(regex='_SPEC')))]

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

    def calculate_decile_intervals_all_columns(SELF, df, decimals=3):
        intervals = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            # Exclure les NaN pour le calcul
            valid_data = df[column].dropna()
            if not valid_data.empty:
                deciles = np.percentile(valid_data, q=np.arange(0, 101, 10))
                # Ajuster les bornes
                deciles[0] = np.floor(
                    deciles[0] * 10 ** decimals) / 10 ** decimals  # Arrondi au-dessous pour le premier
                deciles[-1] = np.ceil(
                    deciles[-1] * 10 ** decimals) / 10 ** decimals  # Arrondi au-dessus pour le dernier

                # Créer les intervalles avec arrondi
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

        self.df = self.df.map(lambda x: None if pd.isna(x) else x)
        self.chemical_subset = self.chemical_subset.map(lambda x: None if pd.isna(x) else x)

        # Ensure correct data types
        self.df = self.df.convert_dtypes()
        self.chemical_subset = self.chemical_subset.convert_dtypes()

        # Replace pd.NA with None
        self.df = self.df.where(self.df.notna(), None)
        self.chemical_subset = self.chemical_subset.where(self.chemical_subset.notna(), None)

    def create_orm_objects(self):
        """
        Create ORM objects from the processed DataFrame.
        """
        # Create ORM instances for stars
        for _, row in self.df.iterrows():
            star = DatabaseTables.Star(
                apogee_id=row['APOGEE_ID'],
                name=row.get('APOGEE_ID'),  # Assuming name is the same as APOGEE_ID
                nb_visits=self.replace_na(row.get('NVISITS')),
                ak_wise=self.replace_na(row.get('AK_WISE')),
                snr=self.replace_na(row.get('SNR')),
                m_h=self.replace_na(row.get('M_H')),
                m_h_err=self.replace_na(row.get('M_H_ERR')),
                vsini=self.replace_na(row.get('VSINI')),
                vmicro=self.replace_na(row.get('VMICRO')),
                vmacro=self.replace_na(row.get('VMACRO')),
                teff=self.replace_na(row.get('TEFF')),
                teff_err=self.replace_na(row.get('TEFF_ERR')),
                logg=self.replace_na(row.get('LOGG')),
                logg_err=self.replace_na(row.get('LOGG_ERR')),
                j=self.replace_na(row.get('J')),
                j_err=self.replace_na(row.get('J_ERR')),
                h=self.replace_na(row.get('H')),
                h_err=self.replace_na(row.get('H_ERR')),
                k=self.replace_na(row.get('K')),
                k_err=self.replace_na(row.get('K_ERR')),
            )
            # Handle foreign key relationships for ranges
            star.m_h_id = self.get_range_id('metalicity', star.m_h)
            star.teff_id = self.get_range_id('temperature', star.teff)
            star.logg_id = self.get_range_id('surface_gravity', star.logg)
            star.vsini_id = self.get_range_id('vsini', star.vsini)
            star.vmicro_id = self.get_range_id('vmicro', star.vmicro)
            star.vmacro_id = self.get_range_id('vmacro', star.vmacro)
            star.j_err_id = self.get_range_id('j_error', star.j_err)
            star.h_err_id = self.get_range_id('h_error', star.h_err)
            star.k_err_id = self.get_range_id('k_error', star.k_err)

            print(star.m_h_id)
            print(star.m_h)

            self.star_objects.append(star)

        # Create ORM instances for ranges
        self.create_range_objects()

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

    def display_current_state(self):
        """
        Display the current state of processed datasets.
        """
        print("Main Dataset Columns:")
        print(self.df.columns)

        print("Chemical Subset Columns:")
        print(self.chemical_subset.columns)

        print("Chemical Error Columns:")
        print(self.chemical_subset_err.columns)

        print("Chemical Flag Columns:")
        print(self.chemical_subset_flag.columns)

        print("Parsed Chemical Elements:")
        print(self.chemical_elements)

        print("Decile Intervals:")
        print(self.decile_intervals)

        print("Decile Intervals for Chemical Subset:")
        print(self.decile_intervals_chemical)

    def insert_into_database(self):
        """
        Insert ORM objects into the database using the database connector.

        This method uses the database connector to obtain a session and adds the ORM objects
        to the session, committing the transaction to insert the data into the database.
        """
        try:
            with self.connector.session_scope() as session:
                # Add range objects first to ensure foreign key constraints are met
                for obj in self.range_objects:
                    session.merge(obj)  # Use merge to avoid duplicates

                # Add star objects
                for star in self.star_objects:
                    session.merge(star)  # Use merge to avoid duplicates

                print("Data inserted successfully!")
        except Exception as e:
            print(f"An error occurred during database insertion: {e}")

    def concatenate_datasets(self, other_df: pd.DataFrame):
        """
        Concatenate another DataFrame to the main DataFrame.

        Parameters:
            other_df (pd.DataFrame): The DataFrame to concatenate with the main DataFrame.

        This method merges the provided DataFrame with the main DataFrame along the appropriate axis,
        ensuring that the data is aligned correctly for further processing.
        """
        self.df = pd.concat([self.df, other_df], ignore_index=True)

    def run(self):
        """
        Run the entire pipeline.

        This method orchestrates the data processing steps: preprocessing, cleaning,
        generating ranges, preparing for insertion, creating ORM objects, and inserting
        the data into the database.
        """
        self.preprocess()
        self.filter_and_clean()
        self.generate_ranges()
        self.prepare_for_insertion()
        self.create_orm_objects()
        self.insert_into_database()



# Example Usage
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('../../data/allStarLite_10rows.csv')

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

    # Display current state
    #pipeline.display_current_state()

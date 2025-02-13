import time
import numpy as np
import pandas as pd
from math import ceil

from tqdm import tqdm

from pipeline.const import table_column_mapping, to_remove_from_main_dataset, to_remove_from_chemicals, CHEMICAL_ELEMENTS_DATA
from database.creation import DatabaseTables

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
        self.decile_intervals_chemical_err = None

        self.surveys = None
        self.telescopes = None
        self.survey_name_to_obj = {}
        self.telescope_name_to_obj = {}
        self.chemical_flag_name_to_obj = {}

        self.symbol_to_name = {data['symbol'].upper(): data['name'] for data in CHEMICAL_ELEMENTS_DATA.values()}


        self.flag_list_columns = None
        self.all_flags = None
        self.chemical_flag_columns = None
        self.all_chemical_flags = None
        self.chemical_flag_list_columns = None


        self.flag_columns = None
        self.all_fields = None


        self.star_objects = []
        self.range_objects = []
        self.fields_objects = []

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

        if 'FIELD' in df.columns:
            # Split the 'FIELD' column into lists
            df['FIELD_LIST'] = df['FIELD'].apply(lambda x: str(x).split(',') if pd.notnull(x) else [])
            # Get all unique fields
            all_fields = set()
            df['FIELD_LIST'].apply(lambda x: all_fields.update(x))
            self.fields = pd.DataFrame({
                'field_name': list(all_fields)
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

        #identify all columns that contains "flag" in in their name in the main dataset (could be starflags)
        flag_columns = df.filter(regex='FLAG', axis=1).columns
        self.flag_columns = flag_columns

        self.all_flags = set()

        for flag_col in flag_columns:
            df[flag_col + '_LIST'] = df[flag_col].apply(lambda x: str(x).split(',') if pd.notnull(x) else [])
            df[flag_col + '_LIST'].apply(lambda x: self.all_flags.update([flag.strip() for flag in x if flag.strip()]))

        self.flag_list_columns = [flag_col + '_LIST' for flag_col in flag_columns]

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
        self.chemical_elements = ['APOGEE_ID' if x == 'APOGEE' else x for x in self.chemical_elements]

        # Process chemical flags
        self.chemical_flag_columns = self.chemical_subset_flag.columns.tolist()
        self.all_chemical_flags = set()

        # For each chemical flag column
        for flag_col in self.chemical_flag_columns:
            # For each row, parse the flags
            self.chemical_subset_flag[flag_col + '_LIST'] = self.chemical_subset_flag[flag_col].apply(
                lambda x: str(x).split(',') if pd.notnull(x) else [])
            # Update the set of all chemical flags
            self.chemical_subset_flag[flag_col + '_LIST'].apply(
                lambda x: self.all_chemical_flags.update([flag.strip() for flag in x if flag.strip()]))

        # Store the list columns for later use
        self.chemical_flag_list_columns = [col + '_LIST' for col in self.chemical_flag_columns]



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
        self.decile_intervals_chemical_err = self.calculate_decile_intervals_all_columns(self.chemical_subset_err)


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

    def create_starflag_objects(self):
        """
        Create ORM objects for starflags and insert them into the database.
        """
        if self.all_flags:
            starflag_objects = []
            for flag_name in self.all_flags:
                starflag = DatabaseTables.Starflag(name=flag_name)
                starflag_objects.append(starflag)
            try:
                with self.connector.session_scope() as session:
                    session.bulk_save_objects(starflag_objects)
                    session.commit()
                    starflags_in_db = session.query(DatabaseTables.Starflag).all()
                    self.starflag_name_to_obj = {starflag.name: starflag for starflag in starflags_in_db}
            except Exception as e:
                print(f"An error occurred during starflag insertion: {e}")

    def create_chemical_flag_objects(self):
        """
        Create ORM objects for chemical flags and insert them into the database.
        """
        if self.all_chemical_flags:
            chemical_flag_objects = []
            for flag_name in self.all_chemical_flags:
                chemical_flag = DatabaseTables.ChemicalFlag(name=flag_name)
                chemical_flag_objects.append(chemical_flag)
            try:
                with self.connector.session_scope() as session:
                    session.bulk_save_objects(chemical_flag_objects)
                    session.commit()
                    chemical_flags_in_db = session.query(DatabaseTables.ChemicalFlag).all()
                    self.chemical_flag_name_to_obj = {flag.name: flag for flag in chemical_flags_in_db}
            except Exception as e:
                print(f"An error occurred during chemical flag insertion: {e}")

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


    def create_fields_objects(self):
        """
        Create ORM objects for fields and insert them into the database.
        """
        print("insert fields")

        if self.fields is not None:
            field_objects = []
            for idx, row in self.fields.iterrows():
                field = DatabaseTables.Field(field_name=row['field_name'])
                field_objects.append(field)
            try:
                with self.connector.session_scope() as session:
                    session.bulk_save_objects(field_objects)
                    session.commit()
                    # Build mapping from field name to ORM object
                    fields_in_db = session.query(DatabaseTables.Field).all()
                    self.field_name_to_obj = {field.field_name: field for field in fields_in_db}
            except Exception as e:
                print(f"An error occurred during field insertion: {e}")
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

                # Associate flags
                for flag_list_col in self.flag_list_columns:
                    flag_list = getattr(row, flag_list_col, [])
                    for flag_name in flag_list:
                        flag_name = flag_name.strip()
                        starflag_obj = self.starflag_name_to_obj.get(flag_name)
                        if starflag_obj:
                            star.starflags.append(starflag_obj)

                ra = self.replace_na(getattr(row, 'RA', None))
                dec = self.replace_na(getattr(row, 'DEC', None))
                field_list = getattr(row, 'FIELD_LIST', [])
                field_name = field_list[0].strip() if field_list else None

                if field_name:
                    field_obj = self.field_name_to_obj.get(field_name)
                else:
                    field_obj = None

                coordinates = DatabaseTables.Coordinates(
                    ra=ra,
                    dec=dec,
                    field=field_obj
                )

                # Assignation des coordonnées à l'étoile
                star.coordinates = coordinates

                star_objects.append(star)
                count += 1

            # Insert batch into database
            self.insert_batch_into_database(star_objects)

        print("Number of stars:", count)

    def create_chemical_abundance_objects(self, batch_size=1000):
        """
        Create ORM objects for chemical abundances and insert them into the database in batches.
        """
        total_rows = self.chemical_subset.shape[0]
        num_batches = ceil(total_rows / batch_size)
        count = 0

        for batch_num in tqdm(range(num_batches), desc="Processing chemical abundance batches"):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_rows)
            batch_df = self.chemical_subset.iloc[start_idx:end_idx]
            batch_df_err = self.chemical_subset_err.iloc[start_idx:end_idx]
            batch_df_flag = self.chemical_subset_flag.iloc[start_idx:end_idx]

            chemical_abundance_objects = []

            for idx, row in batch_df.iterrows():
                apogee_id = row['APOGEE_ID']
                # For each element
                for element in self.chemical_elements:
                    if element == 'APOGEE_ID':
                        continue
                    # Determine the correct abundance column
                    if f"{element}_FE" in batch_df.columns:
                        column_name = f"{element}_FE"
                        error_column = f"{element}_FE_ERR"
                        flag_col = f"{element}_FE_FLAG_LIST"
                    elif f"{element}_H" in batch_df.columns:
                        column_name = f"{element}_H"
                        error_column = f"{element}_H_ERR"
                        flag_col = f"{element}_H_FLAG_LIST"
                    else:
                        continue  # Skip if neither column exists

                    abundance_value = self.replace_na(row.get(column_name, None))
                    error_value = self.replace_na(
                        batch_df_err.at[idx, error_column]) if error_column in batch_df_err.columns else None

                    # Get the value_id and error_id from the ranges
                    value_id = self.get_chemical_range_id(element, abundance_value)
                    error_id = self.get_chemical_error_range_id(element, error_value)

                    # Map element symbol to full name
                    element_full_name = self.symbol_to_name.get(element.upper(), element)

                    # Create ChemicalAbundance object
                    chemical_abundance = DatabaseTables.ChemicalAbundance(
                        apogee_id=apogee_id,
                        element_name=element_full_name,
                        value=abundance_value,
                        value_id=value_id,
                        error=error_value,
                        error_id=error_id
                    )

                    # Associate flags
                    flag_list = batch_df_flag.at[idx, flag_col] if flag_col in batch_df_flag.columns else []
                    for flag_name in flag_list:
                        flag_name = flag_name.strip()
                        chemical_flag_obj = self.chemical_flag_name_to_obj.get(flag_name)
                        if chemical_flag_obj:
                            chemical_abundance.flags.append(chemical_flag_obj)

                    chemical_abundance_objects.append(chemical_abundance)
                    count += 1

            # Insert batch into database
            self.insert_chemical_abundance_batch(chemical_abundance_objects)

        print("Number of chemical abundances:", count)

    def insert_chemical_abundance_batch(self, chemical_abundance_objects):
        """
        Insert a batch of ChemicalAbundance ORM objects into the database.
        """
        try:
            with self.connector.session_scope() as session:
                session.add_all(chemical_abundance_objects)
                session.commit()
        except Exception as e:
            print(f"An error occurred during chemical abundance database insertion: {e}")

    def get_chemical_range_id(self, element_symbol, value):
        """
        Get the range ID for a given chemical abundance value based on decile intervals.
        """
        if value is None or pd.isna(value):
            return None

        # Construct the range column name
        range_col = f"{element_symbol}_FE_RANGE" if f"{element_symbol}_FE_RANGE" in self.decile_intervals_chemical.columns else f"{element_symbol}_H_RANGE"

        if range_col not in self.decile_intervals_chemical.columns:
            return None

        intervals_series = self.decile_intervals_chemical[range_col]

        # Iterate over the intervals
        for idx, interval in intervals_series.items():
            if pd.isna(interval):
                continue
            lower, upper = interval
            if lower <= value <= upper:
                return f"{element_symbol}_{idx}"
        return None

    def get_chemical_error_range_id(self, element_symbol, error_value):
        """
        Get the error range ID for a given chemical abundance error value based on decile intervals.
        """
        if error_value is None or pd.isna(error_value):
            return None

        # Construct the possible range column names
        possible_range_cols = [
            f"{element_symbol}_FE_ERR_RANGE",
            f"{element_symbol}_H_ERR_RANGE"
        ]

        # Find the correct range column
        range_col = next((col for col in possible_range_cols if col in self.decile_intervals_chemical_err.columns),
                         None)

        if not range_col:
            return None

        intervals_series = self.decile_intervals_chemical_err[range_col]

        # Iterate over the intervals
        for idx, interval in intervals_series.items():
            if pd.isna(interval):
                continue
            lower, upper = interval
            if lower <= error_value <= upper:
                return f"{element_symbol}_ERR_{idx}"
        return None

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

            # Create ranges for chemical element errors
        for column in self.chemical_subset_err.columns:
            element_name = column.split('_')[0]  # Get the element name from the column name
            intervals = self.decile_intervals_chemical_err.get(f"{column}_RANGE")
            if intervals is not None:
                for idx, (lower, upper) in enumerate(intervals):
                    range_id = f"{element_name}_ERR_{idx}"
                    range_instance = DatabaseTables.ChemicalErrorRange(
                        id=range_id,
                        element_name=element_name,
                        error_range=f"{lower} to {upper}"
                    )
                    self.range_objects.append(range_instance)

        symbol_to_name = {data['symbol'].upper(): data['name'] for data in CHEMICAL_ELEMENTS_DATA.values()}

        # Create ranges for chemical element errors
        for column in self.chemical_subset_err.columns:
            element_symbol = column.split('_')[0]  # Get the element symbol
            element_name = symbol_to_name.get(element_symbol.upper(), element_symbol)
            intervals = self.decile_intervals_chemical_err.get(f"{column}_RANGE")
            if intervals is not None:
                for idx, (lower, upper) in enumerate(intervals):
                    range_id = f"{element_symbol}_ERR_{idx}"
                    range_instance = DatabaseTables.ChemicalErrorRange(
                        id=range_id,
                        element_name=element_name,  # Use full name
                        error_range=f"{lower} to {upper}"
                    )
                    self.range_objects.append(range_instance)

        # Create ranges for chemical elements
        for column in self.chemical_subset.columns:
            if column == 'APOGEE_ID':
                continue
            element_symbol = column.split('_')[0]
            element_name = symbol_to_name.get(element_symbol.upper(), element_symbol)
            intervals = self.decile_intervals_chemical.get(f"{column}_RANGE")
            if intervals is not None:
                for idx, (lower, upper) in enumerate(intervals):
                    range_id = f"{element_symbol}_{idx}"
                    range_instance = DatabaseTables.ChemicalElementRange(
                        id=range_id,
                        element_name=element_name,  # Use full name
                        element_range=f"{lower} to {upper}"
                    )
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
        except Exception as e:
            print(f"An error occurred during database insertion: {e}")

    def insert_chemical_elements(self):
        """
        Insert chemical elements into the database.
        """
        chemical_element_objects = []
        for key, data in CHEMICAL_ELEMENTS_DATA.items():
            element = DatabaseTables.ChemicalElement(
                name=data['name'],
                symbol=data['symbol'],
                atomic_number=data['atomic_number'],
                electronic_configuration=data['electronic_configuration'],
                period=data['period'],
                family=data['family'],
            )
            chemical_element_objects.append(element)
        try:
            with self.connector.session_scope() as session:
                session.bulk_save_objects(chemical_element_objects)
                session.commit()
        except Exception as e:
            print(f"An error occurred during chemical element insertion: {e}\n")

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
        print("Preprocessing done\n")
        self.filter_and_clean()
        print("Filtering and cleaning done\n")
        self.generate_ranges()
        print("Ranges generated\n")
        self.create_range_objects()
        print("Range objects created\n")

        self.create_starflag_objects()
        print("Starflag objects created\n")
        self.create_chemical_flag_objects()
        print("Chemical flag objects created \n")
        self.create_fields_objects()
        self.prepare_for_insertion()
        self.inser_telecope_survey()
        self.insert_chemical_elements()
        print("Chemical elements inserted\n")
        self.create_orm_objects()
        print("ORM objects created\n")

        self.create_chemical_abundance_objects()
        #self.insert_into_database()
        print("Insertion done")


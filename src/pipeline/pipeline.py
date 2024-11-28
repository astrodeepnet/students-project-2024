import pandas as pd

class DataPipeline:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data pipeline with the main dataset.
        """
        self.df = df
        self.chemical_subset = None
        self.chemical_subset_err = None
        self.chemical_subset_flag = None
        self.chemical_elements = None

    def preprocess(self):
        """
        Preprocess the dataset: drop rows, filter columns, etc.
        """
        # Drop first row
        self.df = self.df.drop(0)

        # Filter chemical abundances
        self.chemical_subset = self.df.filter(regex='_FE', axis=1)
        self.chemical_subset = pd.concat([self.chemical_subset, self.df.filter(regex='_H', axis=1)], axis=1)

        # Remove specific columns from chemical_subset
        to_remove_from_chemicals = [
            'RV_FEH', 'MIN_H', 'MAX_H', 'GAIAEDR3_R_HI_GEO', 'GAIAEDR3_R_HI_PHOTOGEO', 'CU_FE_ERR',
            'P_FE_ERR', 'P_FE_FLAG', 'CU_FE_FLAG', 'M_H', 'M_H_ERR', 'X_H_SPEC', 'X_H', 'X_H_ERR'
        ]
        self.chemical_subset = self.chemical_subset.drop(to_remove_from_chemicals, axis=1)

        # Create main dataset without chemical abundances
        self.df = self.df.drop(self.chemical_subset.columns, axis=1)

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

    def format_for_db(self):
        """
        Format datasets for database insertion.
        """
        # Example formatting
        star_data = self.df[['APOGEE_ID', 'TELESCOPE', 'FIELD', 'RA', 'DEC']].drop_duplicates()
        chemical_data = self.chemical_subset

        return star_data, chemical_data

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


# Example Usage
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('../../data/allStarLite_10rows.csv')

    # Initialize pipeline
    pipeline = DataPipeline(df)

    # Execute transformations
    pipeline.preprocess()
    pipeline.filter_and_clean()

    # Display current state for debugging
    pipeline.display_current_state()

    # Get data ready for DB insertion
    star_data, chemical_data = pipeline.format_for_db()

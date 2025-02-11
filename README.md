# students-project-2024
Repository for the project of M1 data science students

## Setup

### Install the required packages

```bash
pip install -r requirements.txt
```

### Download the data

Download from the [official_APOGEE website](https://www.sdss4.org/dr17/irspec/spectro_data/) the files:
- allStarLite-dr17-synspec_rev1.fits

Paste the file in the data folder.

### Install postgresql

Go to the [official website](https://www.postgresql.org/download/) and follow the instructions to install postgresql on your machine.

### Create a connection to the database

Edit or create src/database/connection.py file with the following content:

```python
password = "XXXXXXX;"
username = "YYYYYYY" # default is "postgres"
database = "ZZZZZZZ" # default is "postgres"
host = "localhost"
port = 5432

DATABASE_URL = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
```
#### Run the pipeline

```bash
python src/main.py
```

#### Generate dataset for data mining

run the following ipython notebook:
```bash
jupyter notebook notebooks/s2/resample.ipynb
```

It should generated a .csv with all of the features and rows neede for the data mining steps.
Data is :
- normalised
- without missing values
- without outliers
- with the target classes inputed
- TODO : balances the classes or add weight to minority classes

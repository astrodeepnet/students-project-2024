from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

def create_all_tables(engine):
    """
    Create all tables in the database
    :param engine:
    :return:
    """
    Base.metadata.create_all(engine)

def drop_all_tables(engine):
    """
    Drop all tables in the database
    :param engine:
    :return:
    """
    Base.metadata.drop_all(engine)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

class DatabaseConnector:
    def __init__(self, database_url: str):
        """
        Initialize the DatabaseConnector with the given database URL.

        Parameters:
            database_url (str): The database connection URL.
        """
        self.database_url = database_url
        self.engine = self.create_engine()
        self.Session = sessionmaker(bind=self.engine)

    def create_engine(self):
        """
        Create and return a new SQLAlchemy engine connected to the database.

        Returns:
            Engine: A SQLAlchemy Engine instance for database connections.
        """
        return create_engine(self.database_url)

    def create_session(self):
        """
        Create and return a new SQLAlchemy session for database transactions.

        Returns:
            Session: An active SQLAlchemy session.
        """
        return self.Session()

    def close_session(self, session):
        """
        Close the provided SQLAlchemy session.

        Parameters:
            session (Session): The session to close.
        """
        session.close()

    def create_tables(self, base):
        """
        Create all tables in the database defined by the ORM Base.

        Parameters:
            base (DeclarativeMeta): The declarative base containing ORM models.
        """
        base.metadata.create_all(self.engine)

    def drop_tables(self, base):
        """
        Drop all tables in the database defined by the ORM Base.

        Parameters:
            base (DeclarativeMeta): The declarative base containing ORM models.
        """
        base.metadata.drop_all(self.engine)

    def execute_query(self, query):
        """
        Execute a raw SQL query against the database.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            ResultProxy: The result of the executed query.
        """
        with self.engine.connect() as connection:
            result = connection.execute(query)
            return result

    def add_records(self, session, records):
        """
        Add multiple ORM records to the session.

        Parameters:
            session (Session): The active SQLAlchemy session.
            records (list): A list of ORM instances to add.
        """
        session.add_all(records)

    def commit_session(self, session):
        """
        Commit the current transaction in the session.

        Parameters:
            session (Session): The session to commit.
        """
        try:
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise e

    def rollback_session(self, session):
        """
        Roll back the current transaction in the session.

        Parameters:
            session (Session): The session to roll back.
        """
        session.rollback()

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.

        Yields:
            Session: A SQLAlchemy session.
        """
        session = self.create_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_engine_url(self):
        """
        Retrieve the database engine URL.

        Returns:
            str: The database engine URL.
        """
        return str(self.engine.url)

    def test_connection(self):
        """
        Test the database connection by attempting to connect.

        Raises:
            Exception: If the connection cannot be established.
        """
        try:
            with self.engine.connect():
                pass  # Connection successful
        except Exception as e:
            raise Exception(f"Database connection failed: {e}")

    def dispose_engine(self):
        """
        Dispose of the database engine, closing all connections.

        This is useful for cleaning up in applications that use connection pooling.
        """
        self.engine.dispose()


if __name__ == "__main__":
    from connection import DATABASE_URL

    # Assume 'DATABASE_URL' is your database connection string and 'Base' is your declarative base
    connector = DatabaseConnector(DATABASE_URL)

    # Test connection
    try:
        connector.test_connection()
        print("Connection successful.")
    except Exception as e:
        print(e)


from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, declarative_base

class DatabaseTables:

    Base = declarative_base()

    class StarSurveyAssociation(Base):
        __tablename__ = 'star_survey_associations'
        id = Column(Integer, primary_key=True, autoincrement=True)
        star_id = Column(String, ForeignKey('stars.apogee_id'), nullable=False)
        survey_id = Column(Integer, ForeignKey('surveys.id'), nullable=False)


    class StarTelescopeAssociation(Base):
        __tablename__ = 'star_telescope_associations'
        id = Column(Integer, primary_key=True, autoincrement=True)
        star_id = Column(String, ForeignKey('stars.apogee_id'), nullable=False)
        telescope_id = Column(Integer, ForeignKey('telescopes.id'), nullable=False)


    class StarStarflagAssociation(Base):
        __tablename__ = 'star_starflag_associations'
        id = Column(Integer, primary_key=True, autoincrement=True)
        star_id = Column(String, ForeignKey('stars.apogee_id'), nullable=False)
        starflag_id = Column(Integer, ForeignKey('starflags.id'), nullable=False)

    class Star(Base):
        __tablename__ = 'stars'
        apogee_id = Column(String, primary_key=True)
        name = Column(String)

        coordinates_id = Column(Integer, ForeignKey('coordinates.id'))
        coordinates = relationship('Coordinates', back_populates='star', uselist=False)

        surveys = relationship(
            'Survey',
            secondary='star_survey_associations',
            back_populates='stars'
        )

        telescopes = relationship(
            'Telescope',
            secondary='star_telescope_associations',
            back_populates='stars'
        )

        starflags = relationship(
            'Starflag',
            secondary='star_starflag_associations',
            back_populates='stars'
        )

        nb_visits = Column(Integer)
        ak_wise = Column(Float)
        snr = Column(Float)

        m_h_id = Column(String, ForeignKey('metalicity.id'))
        m_h = Column(Float)

        m_h_err = Column(Float)

        vsini_id = Column(String, ForeignKey('vsini.id'))
        vsini = Column(Float)

        vmicro_id = Column(String, ForeignKey('vmicro.id'))
        vmicro = Column(Float)
        vmacro_id = Column(String, ForeignKey('vmacro.id'))
        vmacro = Column(Float)
        teff_id = Column(String, ForeignKey('temperature.id'))
        teff = Column(Float)
        teff_err = Column(Float)
        logg_id = Column(String, ForeignKey('surface_gravity.id'))
        logg = Column(Float)
        logg_err = Column(Float)

        j = Column(Float)
        j_err_id = Column(String, ForeignKey('j_error.id'))
        j_err = Column(Float)
        h = Column(Float)
        h_err_id = Column(String, ForeignKey('h_error.id'))
        h_err = Column(Float)
        k = Column(Float)
        k_err_id = Column(String, ForeignKey('k_error.id'))
        k_err = Column(Float)

    class Survey(Base):
        __tablename__ = 'surveys'
        id = Column(Integer, primary_key=True)
        name = Column(String)

        stars = relationship(
            'Star',
            secondary='star_survey_associations',
            back_populates='surveys'
        )


    class Telescope(Base):
        __tablename__ = 'telescopes'
        id = Column(Integer, primary_key=True)
        name = Column(String)

        stars = relationship(
            'Star',
            secondary='star_telescope_associations',
            back_populates='telescopes'
        )

    class Starflag(Base):
        __tablename__ = 'starflags'
        id = Column(Integer, primary_key=True)
        name = Column(String)
        description = Column(Text)

        stars = relationship(
            'Star',
            secondary='star_starflag_associations',
            back_populates='starflags'
        )

    class Field(Base):
        """field table, id is field_id"""
        __tablename__ = 'field'
        id = Column(Integer, primary_key=True)
        field_name = Column(String)

        coordinates = relationship('Coordinates', back_populates='field', cascade='all, delete-orphan')


    class Coordinates(Base):
        """coordinates table, id is coordinates_id (int autoincremet), ref to field in the field table
        contain right ascention and declination"""
        __tablename__ = 'coordinates'
        id = Column(Integer, primary_key=True)
        field_id = Column(Integer, ForeignKey('field.id'))
        ra = Column(Float)
        dec = Column(Float)

        field = relationship('Field', back_populates='coordinates')
        star = relationship('Star', back_populates='coordinates', uselist=False)

    class J_error(Base):
        """J_error table, id is the j_error (string) and contain j_err_range"""
        __tablename__ = 'j_error'
        id = Column(String, primary_key=True)
        j_err_range = Column(String)

    class K_error(Base):
        """K_error table, id is the k_error (string) and contain k_err_range"""
        __tablename__ = 'k_error'
        id = Column(String, primary_key=True)
        k_err_range = Column(String)

    class H_error(Base):
        """H_error table, id is the h_error (string) and contain h_err_range"""
        __tablename__ = 'h_error'
        id = Column(String, primary_key=True)
        h_err_range = Column(String)

    class Metalicity(Base):
        """Metalicity table, id is the metalicity (string) and contain metalicity_range"""
        __tablename__ = 'metalicity'
        id = Column(String, primary_key=True)
        m_h_range = Column(String)

    class Temperature(Base):
        """Temperature table, id is the temperature (string) and contain temperature_range"""
        __tablename__ = 'temperature'
        id = Column(String, primary_key=True)
        teff_range = Column(String)

    class Surface_Gravity(Base):
        """Surface_Gravity table, id is the surface_gravity (string) and contain surface_gravity_range"""
        __tablename__ = 'surface_gravity'
        id = Column(String, primary_key=True)
        logg_range = Column(String)

    class Vmicro(Base):
        """Vmicro table, id is the Vmicro (string) and contain Vmicro_range"""
        __tablename__ = 'vmicro'
        id = Column(String, primary_key=True)
        vmicro_range = Column(String)

    class Vmacro(Base):
        """Vmacro table, id is the Vmacro (string) and contain Vmacro_range"""
        __tablename__ = 'vmacro'
        id = Column(String, primary_key=True)
        vmacro_range = Column(String)

    class Vsini(Base):
        """Vsini table, id is the Vsini (string) and contain Vsini_range"""
        __tablename__ = 'vsini'
        id = Column(String, primary_key=True)
        vsini_range = Column(String)



    ###### CHEMICAL ABUNDANCE TABLES - tables for 2nd DFM schema ######

    class ChemicalErrorRange(Base):
        __tablename__ = 'chemical_error_range'
        id = Column(String, primary_key=True)
        element_name = Column(String)
        error_range = Column(String)

    class ChemicalElementRange(Base):
        __tablename__ = 'chemical_element_range'
        id = Column(String, primary_key=True)
        element_name = Column(String)
        element_range = Column(String)

    class ChemicalElement(Base):
        __tablename__ = 'chemical_element'
        name = Column(String, primary_key=True)
        symbol = Column(String)
        atomic_number = Column(Integer)
        electronic_configuration = Column(String)
        period = Column(Integer)
        family = Column(String)

    class ChemicalAbundance(Base):
        __tablename__ = 'chemical_abundance'
        id = Column(Integer, primary_key=True)
        apogee_id = Column(String, ForeignKey('stars.apogee_id'), nullable=False)
        element_name = Column(String, ForeignKey('chemical_element.name'), nullable=False)
        value_id = Column(String, ForeignKey('chemical_element_range.id'))
        value = Column(Float)
        error_id = Column(String, ForeignKey('chemical_error_range.id'))
        error = Column(Float)

        flags = relationship('ChemicalFlag',
                             secondary='chemical_flag_association',
                             back_populates='chemical_abundances')

    class ChemicalFlag(Base):
        __tablename__ = 'chemical_flag'
        flag_id = Column(Integer, primary_key=True)
        name = Column(String)
        description = Column(Text)

        chemical_abundances = relationship('ChemicalAbundance',
                                           secondary='chemical_flag_association',
                                           back_populates='flags')

    class ChemicalFlagAssociation(Base):
        __tablename__ = 'chemical_flag_association'
        id = Column(Integer, primary_key=True, autoincrement=True)
        chemical_id = Column(Integer, ForeignKey('chemical_abundance.id'), nullable=False)
        flag_id = Column(Integer, ForeignKey('chemical_flag.flag_id'), nullable=False)

    def get_base(self):
        return self.Base

# Adapted test_insert function
def test_insert(connector):
    try:
        with connector.session_scope() as session:
            # Create records for each table

            # Add a field
            field = DatabaseTables.Field(field_name='Field 1')
            session.add(field)

            # Add coordinates linked to the field
            coord1 = DatabaseTables.Coordinates(ra=83.822, dec=-5.391, field=field)
            coord2 = DatabaseTables.Coordinates(ra=83.633, dec=-5.275, field=field)
            session.add_all([coord1, coord2])

            # Add J, K, and H errors
            j_error = DatabaseTables.J_error(id='J_ERR_1', j_err_range='0.01-0.02')
            k_error = DatabaseTables.K_error(id='K_ERR_1', k_err_range='0.02-0.03')
            h_error = DatabaseTables.H_error(id='H_ERR_1', h_err_range='0.03-0.04')
            session.add_all([j_error, k_error, h_error])

            # Add a star
            star = DatabaseTables.Star(apogee_id='2M00000001+7523377', name='Star 1')
            session.add(star)

            # Associate the star with a starflag
            starflag = DatabaseTables.Starflag(name='Good', description='This is a good star')
            star.starflags.append(starflag)
            session.add(starflag)

            # Associate the star with a telescope
            telescope = DatabaseTables.Telescope(name='Hubble')
            star.telescopes.append(telescope)
            session.add(telescope)

            # Associate the star with a survey
            survey = DatabaseTables.Survey(name='SDSS')
            star.surveys.append(survey)
            session.add(survey)

            # Add a metallicity range
            metalicity = DatabaseTables.Metalicity(id='M_H_1', m_h_range='-0.5 to +0.5')
            session.add(metalicity)

            # Add temperature, surface gravity, vmicro, vmacro, and vsini ranges
            temperature = DatabaseTables.Temperature(id='TEFF_1', teff_range='3500-5000')
            surface_gravity = DatabaseTables.Surface_Gravity(id='LOGG_1', logg_range='1.0-2.5')
            vmicro = DatabaseTables.Vmicro(id='VMICRO_1', vmicro_range='1.0-1.5')
            vmacro = DatabaseTables.Vmacro(id='VMACRO_1', vmacro_range='2.0-2.5')
            vsini = DatabaseTables.Vsini(id='VSINI_1', vsini_range='5.0-10.0')
            session.add_all([temperature, surface_gravity, vmicro, vmacro, vsini])

            session.commit()

            # Add a chemical element
            element = DatabaseTables.ChemicalElement(
                name='Hydrogen',
                symbol='H',
                atomic_number=1,
                electronic_configuration='1s1',
                period=1,
                family='Non-metal'
            )
            session.add(element)

            # Add a chemical abundance
            chemical_abundance = DatabaseTables.ChemicalAbundance(
                apogee_id='2M00000001+7523377',
                element_name='Hydrogen',
                value=12.0,
                error=0.1
            )
            session.add(chemical_abundance)

            # Add a chemical flag
            chemical_flag = DatabaseTables.ChemicalFlag(
                name='Reliable',
                description='This measurement is reliable'
            )
            session.add(chemical_flag)

            # Associate the chemical abundance with a chemical flag
            chemical_abundance.flags.append(chemical_flag)

            # Add a chemical error range
            chemical_error_range = DatabaseTables.ChemicalErrorRange(id='0.1', error_range='0.05-0.15')
            session.add(chemical_error_range)

            print("Insertion réussie !")

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")


if __name__ == "__main__":
    from operations import DatabaseConnector
    from connection import DATABASE_URL

    connector = DatabaseConnector(DATABASE_URL)

    database_tables = DatabaseTables()

    # Drop all tables
    connector.drop_tables(DatabaseTables.Base)

    # Create all tables
    connector.create_tables(DatabaseTables.Base)

    # Run the test_insert function
    test_insert(connector)

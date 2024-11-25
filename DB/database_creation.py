from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from database_connection import DATABASE_URL

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
    """field table, id is field_id (int autoincremet)"""
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






def test_insert():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a starflag
    starflag = Starflag(name='Bad', description='This is a bad star')
    session.add(starflag)
    session.commit()

    # Create a telescope
    telescope = Telescope(name='Keck')
    session.add(telescope)
    session.commit()

    # Create a survey
    survey = Survey(name='APOGEE')
    session.add(survey)
    session.commit()


    # Create a star
    star = Star(apogee_id='2M00000001+7523377', name='Star 1')
    star.starflags.append(starflag)
    star.telescopes.append(telescope)
    star.surveys.append(survey)
    session.add(star)
    session.commit()

    session.close()


if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)

    # drop cascate all tables
    Base.metadata.drop_all(engine)

    # create all tables in the database
    Base.metadata.create_all(engine)

    test_insert()

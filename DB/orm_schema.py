from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from database_connection import DATABASE_URL

Base = declarative_base()

class Chemical(Base):
    __tablename__ = 'chemical'
    __table_args__ = {'schema': 'db_proj'}
    chemical_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    value = Column(Float)
    flag = Column(String(50))
    error = Column(Float)
    spec = Column(String(255))
    family = Column(String(255))
    period = Column(Integer)
    electronic_configuration = Column(String(255))
    atomic_number = Column(Integer)
    symbol = Column(String(50))


class Coordinates(Base):
    __tablename__ = 'coordinates'
    __table_args__ = {'schema': 'db_proj'}
    coordinates_id = Column(Integer, primary_key=True)
    right_ascension = Column(Float)
    declination = Column(Float)
    field = Column(String(255))


class Survey(Base):
    __tablename__ = 'survey'
    __table_args__ = {'schema': 'db_proj'}
    survey_id = Column(Integer, primary_key=True)
    survey_name = Column(String(255))


class Starflag(Base):
    __tablename__ = 'starflag'
    __table_args__ = {'schema': 'db_proj'}
    starflag_id = Column(Integer, primary_key=True)
    starflag_name = Column(String(255))
    description = Column(Text)


class Telescope(Base):
    __tablename__ = 'telescope'
    __table_args__ = {'schema': 'db_proj'}
    telescope_id = Column(Integer, primary_key=True)
    telescope_name = Column(String(255))


class Photometry(Base):
    __tablename__ = 'photometry'
    __table_args__ = {'schema': 'db_proj'}
    photometry_id = Column(Integer, primary_key=True)
    j = Column(Float)
    h = Column(Float)
    k = Column(Float)
    j_err = Column(Float)
    h_err = Column(Float)
    k_err = Column(Float)


class Star(Base):
    __tablename__ = 'star'
    __table_args__ = {'schema': 'db_proj'}
    apogee_id = Column(Integer, primary_key=True)
    chemical_id = Column(Integer, ForeignKey('db_proj.chemical.chemical_id'))
    telescope_id = Column(Integer, ForeignKey('db_proj.telescope.telescope_id'))
    gaia_id = Column(String(255))
    starflag_id = Column(Integer, ForeignKey('db_proj.starflag.starflag_id'))
    survey_id = Column(Integer, ForeignKey('db_proj.survey.survey_id'))
    coordinates_id = Column(Integer, ForeignKey('db_proj.coordinates.coordinates_id'))
    photometry_id = Column(Integer, ForeignKey('db_proj.photometry.photometry_id'))
    nb_observations = Column(Integer)
    temperature_avg = Column(Float)
    metallicity_avg = Column(Float)
    gravity_avg = Column(Float)
    nb_error_flag = Column(Integer)
    ak_wise = Column(Float)
    vsini = Column(Float)
    vhelio = Column(Float)
    vmicro = Column(Float)
    vmacro = Column(Float)
    alpha_m_err = Column(Float)
    alpha_m = Column(Float)
    snr = Column(Float)
    logg_err = Column(Float)
    logg = Column(Float)
    logg_spec = Column(Float)
    teff = Column(Float)
    teff_spec = Column(Float)
    teff_err = Column(Float)
    m_h = Column(Float)
    m_h_err = Column(Float)

    # Relations
    chemical = relationship('Chemical')
    telescope = relationship('Telescope')
    starflag = relationship('Starflag')
    survey = relationship('Survey')
    coordinates = relationship('Coordinates')
    photometry = relationship('Photometry')





engine = create_engine(DATABASE_URL)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# Exemple d'insertion
def insert_sample_data():
    telescope = Telescope(telescope_name="telescope1")
    session.add(telescope)
    session.commit()
    print("Telescope inserted")

if __name__ == "__main__":
    insert_sample_data()

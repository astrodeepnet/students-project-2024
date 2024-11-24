from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from database_connection import DATABASE_URL

Base = declarative_base()

class Chemical(Base):
    __tablename__ = 'chemical'
    Chemical_ID = Column(Integer, primary_key=True)
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
    Coordinates_ID = Column(Integer, primary_key=True)
    Right_ascension = Column(Float)
    Declination = Column(Float)
    Field = Column(String(255))


class Survey(Base):
    __tablename__ = 'survey'
    Survey_ID = Column(Integer, primary_key=True)
    Valeur = Column(String(255))


class Starflag(Base):
    __tablename__ = 'starflag'
    Starflag_ID = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)


class Telescope(Base):
    __tablename__ = 'telescope'
    Telescope_ID = Column(Integer, primary_key=True)
    name = Column(String(255))


class Photometry(Base):
    __tablename__ = 'photometry'
    Photometry_ID = Column(Integer, primary_key=True)
    J = Column(Float)
    H = Column(Float)
    K = Column(Float)
    J_ERR = Column(Float)
    H_ERR = Column(Float)
    K_ERR = Column(Float)


class Star(Base):
    __tablename__ = 'star'
    APOGEE_ID = Column(Integer, primary_key=True)
    Chemical_ID = Column(Integer, ForeignKey('chemical.Chemical_ID'))
    Telescope_ID = Column(Integer, ForeignKey('telescope.Telescope_ID'))
    GAIA_ID = Column(String(255))
    Starflag_ID = Column(Integer, ForeignKey('starflag.Starflag_ID'))
    Survey_ID = Column(Integer, ForeignKey('survey.Survey_ID'))
    Coordinates_ID = Column(Integer, ForeignKey('coordinates.Coordinates_ID'))
    Photometry_ID = Column(Integer, ForeignKey('photometry.Photometry_ID'))
    Nb_Observations = Column(Integer)
    Temperature_AVG = Column(Float)
    Metallicity_AVG = Column(Float)
    Gravity_AVG = Column(Float)
    Nb_Error_Flag = Column(Integer)
    AK_WISE = Column(Float)
    VSINI = Column(Float)
    VHELIO = Column(Float)
    VMICRO = Column(Float)
    VMACRO = Column(Float)
    ALPHA_M_ERR = Column(Float)
    ALPHA_M = Column(Float)
    SNR = Column(Float)
    LOGG_ERR = Column(Float)
    LOGG = Column(Float)
    LOGG_SPEC = Column(Float)
    TEFF = Column(Float)
    TEFF_SPEC = Column(Float)
    TEFF_ERR = Column(Float)
    M_H = Column(Float)
    M_H_ERR = Column(Float)

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

def insert_sample_data():
    chemical = Chemical(
        name="Hydrogen",
        value=0.74,
        flag="H",
        error=0.01,
        spec="H",
        family="Non-metal",
        period=1,
        electronic_configuration="1s1",
        atomic_number=1,
        symbol="H"
    )
    session.add(chemical)
    session.commit()
    print("Sample data inserted into Chemical table.")

if __name__ == "__main__":
    insert_sample_data()

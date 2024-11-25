
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, Table
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from database.database_connection import DATABASE_URL

Base = declarative_base()


star_survey_association = Table(
    'star_survey',
    Base.metadata,
    Column('star_id', Integer, ForeignKey('db_proj.star.apogee_id'), primary_key=True),
    Column('survey_id', Integer, ForeignKey('db_proj.survey.survey_id'), primary_key=True)
)


star_telescope_association = Table(
    'star_telescope',
    Base.metadata,
    Column('star_id', Integer, ForeignKey('db_proj.star.apogee_id'), primary_key=True),
    Column('telescope_id', Integer, ForeignKey('db_proj.telescope.telescope_id'), primary_key=True),
    schema='db_proj'
)


class Star(Base):
    __tablename__ = 'star'
    __table_args__ = {'schema': 'db_proj'}
    apogee_id = Column("apogee_id", Integer, primary_key=True)

    chemical_id = Column("chemical_id", Integer, ForeignKey('db_proj.chemical.chemical_id'))
    starflag_id = Column("starflag_id", Integer, ForeignKey('db_proj.starflag.starflag_id'))
    coordinates_id = Column("coordinates_id", Integer, ForeignKey('db_proj.coordinates.coordinates_id'))
    photometry_id = Column("photometry_id", Integer, ForeignKey('db_proj.photometry.photometry_id'))

    surveys = relationship(
        'Survey',
        secondary=star_survey_association,
        back_populates='stars')

    telescope = relationship(
        'Telescope',
        secondary=star_telescope_association,
        back_populates='stars')




    nb_observations = Column("nb_observations", Integer)
    temperature_avg = Column("temperature_avg", Float)
    metallicity_avg = Column("metallicity_avg", Float)
    gravity_avg = Column("gravity_avg", Float)
    nb_error_flag = Column("nb_error_flag", Integer)
    ak_wise = Column("ak_wise", Float)
    vsini = Column("vsini", Float)
    vhelio = Column("vhelio", Float)
    vmicro = Column("vmicro", Float)
    vmacro = Column("vmacro", Float)
    alpha_m_err = Column("alpha_m_err", Float)
    alpha_m = Column("alpha_m", Float)
    snr = Column("snr", Float)
    logg_err = Column("logg_err", Float)
    logg = Column("logg", Float)
    logg_spec = Column("logg_spec", Float)
    teff = Column("teff", Float)
    teff_spec = Column("teff_spec", Float)
    teff_err = Column("teff_err", Float)
    m_h = Column("m_h", Float)
    m_h_err = Column("m_h_err", Float)

    # Relations
    chemical = relationship('Chemical')
    telescope = relationship('Telescope')
    starflag = relationship('Starflag')
    survey = relationship('Survey')
    coordinates = relationship('Coordinates')
    photometry = relationship('Photometry')


class Survey(Base):
    __tablename__ = 'survey'
    __table_args__ = {'schema': 'db_proj'}
    survey_id = Column("survey_id", Integer, primary_key=True)
    survey_name = Column("survey_name", String(255))

    stars = relationship(
        'Star',
        secondary='db_proj.star_survey',
        back_populates='surveys',
        primaryjoin="Survey.survey_id == StarSurvey.survey_id"
    )


class Telescope(Base):
    __tablename__ = 'telescope'
    __table_args__ = {'schema': 'db_proj'}
    telescope_id = Column("telescope_id", Integer, primary_key=True)
    telescope_name = Column("telescope_name", String(255))

    stars = relationship(
        'Star',
        secondary=star_telescope_association,
        back_populates='telescopes'
    )



class Chemical(Base):
    __tablename__ = 'chemical'
    __table_args__ = {'schema': 'db_proj'}
    chemical_id = Column("chemical_id", Integer, primary_key=True)
    name = Column("name", String(255))
    value = Column("value", Float)
    flag = Column("flag", String(255))
    error = Column("error", Float)
    spec = Column("spec", String(255))
    family = Column("family", String(255))
    period = Column("period", Integer)
    electronic_configuration = Column("electronic_configuration", String(255))
    atomic_number = Column("atomic_number", Integer)
    symbol = Column("symbol", String(50))


class Coordinates(Base):
    __tablename__ = 'coordinates'
    __table_args__ = {'schema': 'db_proj'}
    coordinates_id = Column("coordinates_id", Integer, primary_key=True)
    right_ascension = Column("right_ascension", Float)
    declination = Column("declination", Float)
    field = Column("field", String(255))


class Starflag(Base):
    __tablename__ = 'starflag'
    __table_args__ = {'schema': 'db_proj'}
    starflag_id = Column("starflag_id", Integer, primary_key=True)
    starflag_name = Column("starflag_name", String(255))
    description = Column("description", Text)


class Photometry(Base):
    __tablename__ = 'photometry'
    __table_args__ = {'schema': 'db_proj'}
    photometry_id = Column("photometry_id", Integer, primary_key=True)
    j = Column("j", Float)
    h = Column("h", Float)
    k = Column("k", Float)
    j_err = Column("j_err", Float)
    h_err = Column("h_err", Float)
    k_err = Column("k_err", Float)


# Connexion et création des tables
engine = create_engine(DATABASE_URL)

#Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()


def test_star_survey_relationship():
    star1 = Star(apogee_id=1, nb_observations=10, temperature_avg=5500.0)
    star2 = Star(apogee_id=2, nb_observations=5, temperature_avg=4500.0)

    survey1 = Survey(survey_id=1, survey_name="Survey A")
    survey2 = Survey(survey_id=2, survey_name="Survey B")

    star1.surveys.append(survey1)
    star1.surveys.append(survey2)
    star2.surveys.append(survey1)

    session.add_all([star1, star2, survey1, survey2])
    session.commit()

    print(f"Star {star1.apogee_id} participates in surveys: {[s.survey_name for s in star1.surveys]}")
    print(f"Star {star2.apogee_id} participates in surveys: {[s.survey_name for s in star2.surveys]}")
    print(f"Survey {survey1.survey_name} includes stars: {[s.apogee_id for s in survey1.stars]}")
    print(f"Survey {survey2.survey_name} includes stars: {[s.apogee_id for s in survey2.stars]}")

if __name__ == "__main__":
    test_star_survey_relationship()


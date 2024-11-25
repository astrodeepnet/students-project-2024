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

CREATE TABLE Chemical (
    Chemical_ID INT PRIMARY KEY,
    name VARCHAR(255),
    value FLOAT,
    flag VARCHAR(255),
    error FLOAT,
    spec VARCHAR(255),
    family VARCHAR(255),
    period INT,
    electronic_configuration VARCHAR(255),
    atomic_number INT,
    symbol VARCHAR(50)
);

CREATE TABLE Coordinates (
    Coordinates_ID INT PRIMARY KEY,
    Right_ascension FLOAT,
    Declination FLOAT,
    Field VARCHAR(255)
);

CREATE TABLE Survey (
    Survey_ID INT PRIMARY KEY,
    survey_name VARCHAR(255)
);

CREATE TABLE Starflag (
    Starflag_ID INT PRIMARY KEY,
    starflag_name VARCHAR(255),
    description TEXT
);

CREATE TABLE Telescope (
    Telescope_ID INT PRIMARY KEY,
    telescope_name VARCHAR(255)
);

CREATE TABLE Photometry (
    Photometry_ID INT PRIMARY KEY,
    J FLOAT,
    H FLOAT,
    K FLOAT,
    J_ERR FLOAT,
    H_ERR FLOAT,
    K_ERR FLOAT
);

CREATE TABLE Star (
    APOGEE_ID VARCHAR(255) PRIMARY KEY,
    Chemical_ID INT,
    Telescope_ID INT,
    Starflag_ID INT,
    Survey_ID INT,
    Coordinates_ID INT,
    Photometry_ID INT,
    Nb_visits INT,
    AK_WISE FLOAT,
    VSINI FLOAT,
    VHELIO FLOAT,
    VMICRO FLOAT,
    VMACRO FLOAT,
    ALPHA_M_ERR FLOAT,
    ALPHA_M FLOAT,
    SNR FLOAT,
    LOGG_ERR FLOAT,
    LOGG FLOAT,
    LOGG_SPEC FLOAT,
    TEFF FLOAT,
    TEFF_SPEC FLOAT,
    TEFF_ERR FLOAT,
    M_H FLOAT,
    M_H_ERR FLOAT,

    -- Mesure, TODO add the rest
    Nb_Observations INT,                -- Nombre total d'observations pour l'étoile
    Temperature_AVG FLOAT,              -- Température moyenne
    Metallicity_AVG FLOAT,              -- Métallicité moyenne
    Gravity_AVG FLOAT,                  -- Gravité moyenne
    Nb_Error_Flag INT,                 -- Nombre total de flags d'erreurs associés

    FOREIGN KEY (Chemical_ID) REFERENCES Chemical(Chemical_ID),
    FOREIGN KEY (Telescope_ID) REFERENCES Telescope(Telescope_ID),
    FOREIGN KEY (Starflag_ID) REFERENCES Starflag(Starflag_ID),
    FOREIGN KEY (Survey_ID) REFERENCES Survey(Survey_ID),
    FOREIGN KEY (Coordinates_ID) REFERENCES Coordinates(Coordinates_ID),
    FOREIGN KEY (Photometry_ID) REFERENCES Photometry(Photometry_ID)
);

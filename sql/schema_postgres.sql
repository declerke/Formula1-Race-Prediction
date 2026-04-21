-- F1 Race Prediction — PostgreSQL schema
-- All tables use ON CONFLICT DO NOTHING for idempotent upserts.

CREATE TABLE IF NOT EXISTS drivers (
    driver_id       VARCHAR(64) PRIMARY KEY,
    code            VARCHAR(8),
    given_name      VARCHAR(64)  NOT NULL,
    family_name     VARCHAR(64)  NOT NULL,
    nationality     VARCHAR(64),
    driver_number   INTEGER,
    dob             DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS constructors (
    constructor_id  VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(128) NOT NULL,
    nationality     VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS races (
    race_id         SERIAL PRIMARY KEY,
    season          INTEGER      NOT NULL,
    round           INTEGER      NOT NULL,
    race_name       VARCHAR(128) NOT NULL,
    circuit_id      VARCHAR(64)  NOT NULL,
    circuit_name    VARCHAR(128),
    country         VARCHAR(64),
    locality        VARCHAR(64),
    race_date       DATE,
    race_time       VARCHAR(16),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, round)
);

CREATE TABLE IF NOT EXISTS results (
    result_id           SERIAL PRIMARY KEY,
    season              INTEGER      NOT NULL,
    round               INTEGER      NOT NULL,
    driver_id           VARCHAR(64)  NOT NULL REFERENCES drivers(driver_id),
    constructor_id      VARCHAR(64)  NOT NULL REFERENCES constructors(constructor_id),
    grid                INTEGER,
    finish_position     INTEGER,
    position_text       VARCHAR(8),
    points              NUMERIC(6,2),
    laps                INTEGER,
    status              VARCHAR(64),
    finish_time_ms      BIGINT,
    fastest_lap_rank    INTEGER,
    fastest_lap_time    VARCHAR(16),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, round, driver_id)
);

CREATE TABLE IF NOT EXISTS qualifying (
    qualifying_id       SERIAL PRIMARY KEY,
    season              INTEGER     NOT NULL,
    round               INTEGER     NOT NULL,
    driver_id           VARCHAR(64) NOT NULL REFERENCES drivers(driver_id),
    constructor_id      VARCHAR(64) NOT NULL REFERENCES constructors(constructor_id),
    qualifying_position INTEGER,
    q1_time             VARCHAR(16),
    q2_time             VARCHAR(16),
    q3_time             VARCHAR(16),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, round, driver_id)
);

CREATE TABLE IF NOT EXISTS lap_times (
    lap_time_id     SERIAL PRIMARY KEY,
    session_key     INTEGER,
    season          INTEGER,
    round           INTEGER,
    driver_number   INTEGER,
    driver_id       VARCHAR(64),
    lap_number      INTEGER      NOT NULL,
    lap_duration    NUMERIC(10,4),
    sector1_ms      NUMERIC(10,4),
    sector2_ms      NUMERIC(10,4),
    sector3_ms      NUMERIC(10,4),
    i1_speed        NUMERIC(6,1),
    i2_speed        NUMERIC(6,1),
    st_speed        NUMERIC(6,1),
    is_pit_out_lap  BOOLEAN,
    source          VARCHAR(16)  DEFAULT 'openf1',
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (session_key, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS telemetry (
    telemetry_id    SERIAL PRIMARY KEY,
    session_key     INTEGER      NOT NULL,
    driver_number   INTEGER      NOT NULL,
    stint_number    INTEGER,
    compound        VARCHAR(16),
    tyre_age_at_start INTEGER,
    lap_start       INTEGER,
    lap_end         INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (session_key, driver_number, stint_number)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_results_driver ON results(driver_id);
CREATE INDEX IF NOT EXISTS idx_results_season_round ON results(season, round);
CREATE INDEX IF NOT EXISTS idx_results_constructor ON results(constructor_id);
CREATE INDEX IF NOT EXISTS idx_lap_times_session_driver ON lap_times(session_key, driver_number);
CREATE INDEX IF NOT EXISTS idx_qualifying_season_round ON qualifying(season, round);

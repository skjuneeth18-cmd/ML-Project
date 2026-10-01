-- =========================================================
-- WATER QUALITY CLASSIFICATION - DATABASE SCHEMA (SQLite)
-- =========================================================
-- Three tables:
--   1. water_samples  - the training dataset, so the site can
--                        chart what it was trained on
--   2. predictions    - a log of every prediction made through
--                        the website
--   3. model_metrics  - accuracy/precision/recall/F1 for each
--                        trained model
-- =========================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------
-- 1. water_samples
-- ---------------------------------------------------------
DROP TABLE IF EXISTS water_samples;

CREATE TABLE water_samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    aluminium    REAL NOT NULL,
    ammonia      REAL NOT NULL,
    arsenic      REAL NOT NULL,
    barium       REAL NOT NULL,
    cadmium      REAL NOT NULL,
    chloramine   REAL NOT NULL,
    chromium     REAL NOT NULL,
    copper       REAL NOT NULL,
    flouride     REAL NOT NULL,
    bacteria     REAL NOT NULL,
    viruses      REAL NOT NULL,
    lead         REAL NOT NULL,
    nitrates     REAL NOT NULL,
    nitrites     REAL NOT NULL,
    mercury      REAL NOT NULL,
    perchlorate  REAL NOT NULL,
    radium       REAL NOT NULL,
    selenium     REAL NOT NULL,
    silver       REAL NOT NULL,
    uranium      REAL NOT NULL,
    is_safe      INTEGER NOT NULL CHECK (is_safe IN (0, 1))
);

CREATE INDEX idx_water_samples_is_safe ON water_samples (is_safe);

-- ---------------------------------------------------------
-- 2. predictions
-- ---------------------------------------------------------
DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),

    aluminium    REAL NOT NULL,
    ammonia      REAL NOT NULL,
    arsenic      REAL NOT NULL,
    barium       REAL NOT NULL,
    cadmium      REAL NOT NULL,
    chloramine   REAL NOT NULL,
    chromium     REAL NOT NULL,
    copper       REAL NOT NULL,
    flouride     REAL NOT NULL,
    bacteria     REAL NOT NULL,
    viruses      REAL NOT NULL,
    lead         REAL NOT NULL,
    nitrates     REAL NOT NULL,
    nitrites     REAL NOT NULL,
    mercury      REAL NOT NULL,
    perchlorate  REAL NOT NULL,
    radium       REAL NOT NULL,
    selenium     REAL NOT NULL,
    silver       REAL NOT NULL,
    uranium      REAL NOT NULL,

    svm_prediction     INTEGER NOT NULL CHECK (svm_prediction IN (0, 1)),
    svm_confidence      REAL NOT NULL,
    rf_prediction        INTEGER NOT NULL CHECK (rf_prediction IN (0, 1)),
    rf_confidence         REAL NOT NULL,
    final_prediction     INTEGER NOT NULL CHECK (final_prediction IN (0, 1)),
    models_agree           INTEGER NOT NULL CHECK (models_agree IN (0, 1))
);

CREATE INDEX idx_predictions_created_at ON predictions (created_at);

-- ---------------------------------------------------------
-- 3. model_metrics
-- ---------------------------------------------------------
DROP TABLE IF EXISTS model_metrics;

CREATE TABLE model_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model            TEXT NOT NULL,
    accuracy         REAL NOT NULL,
    precision_score  REAL NOT NULL,
    recall           REAL NOT NULL,
    f1_score         REAL NOT NULL
);

# ME/CFS vs Depression Classification Dataset

This repository provides a **synthetic dataset** designed to explore the differential diagnosis between **Myalgic Encephalomyelitis / Chronic Fatigue Syndrome (ME/CFS)** and **Depression**.  
It is the first open dataset of its kind, created to help beginners and researchers practice machine learning on complex clinical cases.

---

## 🎯 Objective

The goal is to predict whether a patient has:

- **ME/CFS**
- **Depression**
- **Both (comorbidity)**

based on behavioral, clinical, and symptomatic features.

---

## 📋 Dataset Overview

- **Rows:** ~1,000
- **Format:** CSV (UTF-8)
- **Target variable:** `diagnosis`
- **Classes:** `ME/CFS`, `Depression`, `Both`

### Features

| Feature                        | Description                               |
| ------------------------------ | ----------------------------------------- |
| `age`                          | Patient's age                             |
| `gender`                       | Gender (Male / Female / Other)            |
| `fatigue_severity_scale_score` | Fatigue Severity Scale (0–10)             |
| `depression_phq9_score`        | PHQ-9 depression score (0–27)             |
| `pem_present`                  | Post-Exertional Malaise (Yes/No or 1/0)   |
| `pem_duration_hours`           | Duration of PEM (hours)                   |
| `sleep_quality_index`          | Sleep quality (1–10)                      |
| `brain_fog_level`              | Brain fog level (1–10)                    |
| `physical_pain_score`          | Physical pain intensity (1–10)            |
| `stress_level`                 | Stress level (1–10)                       |
| `work_status`                  | Working / Partially working / Not working |
| `social_activity_level`        | Very low – Very high                      |
| `exercise_frequency`           | Never – Daily                             |
| `meditation_or_mindfulness`    | Practices mindfulness/meditation (Yes/No) |
| `hours_of_sleep_per_night`     | Average sleep per night                   |
| `diagnosis`                    | **Target** (ME/CFS, Depression, Both)     |

---

## ⚠️ Key Characteristics

- Contains **1–5% missing values** across most features (to simulate real-world clinical data).
- Controlled **noise** in numeric features prevents perfect separation between classes.
- Diagnosis logic based on **clinical-like heuristics** (not random).

---

## 🛠 Suggested Use Cases

- **Binary classification:** ME/CFS vs Depression
- **Multiclass classification:** ME/CFS, Depression, Both
- **Exploratory Data Analysis (EDA)**
- **Feature engineering** practice
- **Missing data imputation techniques**
- **ML interpretability & explainable AI** in medical datasets

---

## API для класифікації MECFS vs Депресія

Готовий до продакшену FastAPI сервіс для класифікації записів пацієнтів на Депресія, ME/CFS або Обидва. Включає повний конвеєр: попередню обробку, інженерію ознак, навчання моделі (Логістична регресія), висновки та аудит-журнал на базі PostgreSQL для прогнозів та вхідних даних.

### Основні можливості

- Ендпоінти для перевірки здоров'я, прогнозування (пакетне/одиничне), навчання з бази даних та інформації про модель
- Автоматична ініціалізація та наповнення бази даних з `data/processed/feature_engineered_data.csv`
- Журналювання прогнозів (навчання та висновки) з ймовірностями та впевненістю
- Журналювання необроблених HTTP-запитів для відстеження

---

## 1) Структура проекту

```
.
├─ src/
│  ├─ api/main.py                    # FastAPI додаток та ендпоінти
│  ├─ inference_pipeline/inference.py # Конвеєр висновків
│  ├─ feature_pipeline/               # Утиліти попередньої обробки та інженерії ознак
│  └─ db/
│     ├─ models.py                   # SQLAlchemy ORM моделі
│     └─ session.py                  # Двигун + фабрика сесій
├─ data/processed/feature_engineered_data.csv
├─ models/                           # Збережена модель та артефакти
└─ notebooks/                        # Ноутбуки для навчання та EDA
```

---

## 2) Вимоги

- Python 3.11+
- PostgreSQL доступний за адресою:
  - `postgresql+psycopg2://postgres:password@127.0.0.1:5433/me_cfs_vs_depression`
  - Змініть це в `src/db/session.py` за потреби.

Встановіть залежності (рекомендовано: `uv`):

- З `uv` (використовує `pyproject.toml` та `uv.lock`):

```powershell
uv sync
```

- Або встановіть ключові залежності через pip (якщо віддаєте перевагу ручній установці):

```powershell
pip install fastapi uvicorn[standard] sqlalchemy psycopg2-binary pandas numpy scikit-learn joblib
```

---

## 3) Налаштування бази даних

Переконайтеся, що PostgreSQL працює і база даних існує.

Створення БД (приклад з `psql`):

```powershell
psql -h 127.0.0.1 -p 5433 -U postgres -c "CREATE DATABASE me_cfs_vs_depression;"
```

Таблиці, які використовує додаток:

- `predictions`: зберігає прогнози з навчання та висновків з `source` = "train" | "inference"
- `inference_inputs`: зберігає необроблені HTTP-дані для кожного запиту прогнозу
- `features`: основна матриця ознак, використана для навчання найкращої моделі; автоматично створюється при завантаженні CSV

Поведінка при запуску:

- При старті додатку ORM таблиці створюються, якщо їх немає.
- Якщо таблиця `features` відсутня або порожня, додаток автоматично завантажує `data/processed/feature_engineered_data.csv` (з відкатом до `feature_engineered_train.csv`) і додає мітку часу `created_at` 2025 року до кожного рядка.

Ручне завантаження (опціонально):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/load-features
```

---

## 4) Запуск API

```powershell
uvicorn src.api.main:app --reload
```

Відкрийте документацію: `http://localhost:8000/docs`

Перевірка здоров'я:

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health
```

---

## 5) Огляд ендпоінтів

- `GET /` – інформація про API
- `GET /health` – доступність моделі та артефактів
- `POST /predict` – пакетне прогнозування (журналює входи та прогнози в БД)
- `POST /predict_single` – одиничне прогнозування (обгортка пакетного)
- `GET /model_info` – тип моделі, параметри, ознаки
- `POST /load-features` – завантаження `feature_engineered_data.csv` в `features` з мітками часу
- `POST /train-model` – навчання Логістичної регресії на `features` з БД, збереження моделі, журналювання прогнозів навчання

---

## 6) Навчання

Навчіть модель з таблиці `features` (не потрібно тіло JSON):

```powershell
# Розподіл за замовчуванням 90:10, seed=42
Invoke-RestMethod -Method Post -Uri http://localhost:8000/train-model

# Користувацький розподіл та seed
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/train-model?test_size=0.15&random_state=123"
```

Результати:

- Зберігає модель в `models/best_logistic_model.pkl`
- Журналює всі прогнози навчального розподілу в `predictions` з `source="train"`

---

## 7) Висновки (Inference)

Пакетні прогнози (надайте список записів з оригінальними полями схеми як `age`, `gender`, `sleep_quality_index`, тощо):

```powershell
$body = @(
  @{ age = 45; gender = "Female"; sleep_quality_index = 12; brain_fog_level = 7;
     physical_pain_score = 6; stress_level = 5; depression_phq9_score = 8;
     fatigue_severity_scale_score = 40; pem_duration_hours = 24; hours_of_sleep_per_night = 6;
     pem_present = 1; work_status = "Partially working"; social_activity_level = "Low";
     exercise_frequency = "Rarely"; meditation_or_mindfulness = "No" }
)
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body ($body | ConvertTo-Json) -ContentType 'application/json'
```

Одиничне прогнозування:

```powershell
$record = @{ age = 30; gender = "Female"; sleep_quality_index = 9; brain_fog_level = 3;
             physical_pain_score = 2; stress_level = 2; depression_phq9_score = 3;
             fatigue_severity_scale_score = 15; pem_duration_hours = 4; hours_of_sleep_per_night = 8;
             pem_present = 0; work_status = "Not working"; social_activity_level = "Medium";
             exercise_frequency = "Sometimes"; meditation_or_mindfulness = "Yes" }
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict_single -Body ($record | ConvertTo-Json) -ContentType 'application/json'
```

Поведінка:

- Зберігає вхідні дані в `inference_inputs`
- Зберігає кожен прогноз в `predictions` з `source="inference"`

Опціонально: включити інженерні ознаки у відповідь

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict?include_features=true" -Body ($body | ConvertTo-Json) -ContentType 'application/json'
```

---

## 8) Артефакти моделі

Очікуються в `models/`:

- `best_logistic_model.pkl`
- `label_encoder.pkl`
- `robust_scaler.pkl`
- `ordinal_mappings.pkl`
- `imputer_stats.pkl`

`/health` та `/model_info` показують підсумкову інформацію про доступність/деталі.

---

## 9) Ноутбуки

Зверніться до `notebooks/` (наприклад, `02_feature_eng_encoding.ipynb`, `04_modeling.ipynb`) для деталей про те, як було отримано інженерію ознак та параметри Логістичної регресії. Сервіс відображає ці кроки через `src/feature_pipeline` та `src/inference_pipeline`.

---

## 10) Конфігурація

- URL бази даних визначений в `src/db/session.py`. Оновіть його відповідно до вашого середовища за потреби.
- При запуску `features` автоматично заповнюється, якщо порожня. Ви можете повторно заповнити через `POST /load-features`.

---

## 11) Усунення несправностей

- Якщо API запускається, але журналювання БД не працює, прогнози все одно працюють; помилки перехоплюються і не порушують відповіді.
- Переконайтеся, що PostgreSQL доступний і облікові дані правильні.
- Якщо `feature_engineered_data.csv` відсутній, додаток використовує `feature_engineered_train.csv` при заповненні.
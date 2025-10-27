# Homework 5 - Deployment Answers

## Question 1: uv Version
**Command:** `uv --version`

**Answer:** `uv 0.9.5`

---

## Question 2: First Hash for Scikit-Learn 1.6.1
**Commands:**
```bash
mkdir homework5_project && cd homework5_project
uv init
uv add scikit-learn==1.6.1
grep -A 2 "name = \"scikit-learn\"" uv.lock
```

**Answer:** `sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e`

This is the hash for the sdist (source distribution) of scikit-learn 1.6.1.

---

## Question 3: Probability for First Record
**Client:**
```json
{
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
```

**Answer:** `0.534` (closest option: **0.533**)

---

## Question 4: FastAPI Service Probability
**Client:**
```json
{
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
```

**Answer:** `0.534` (closest option: **0.534**)

---

## Question 5: Docker Base Image Size
**Command:** `docker images agrigorev/zoomcamp-model:2025`

**Answer:** `121 MB` (option: **121 MB**)

---

## Question 6: Dockerized Service Probability
**Client:**
```json
{
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
```

**Answer:** `0.99` (closest option: **0.99**)

Note: This uses pipeline_v2.bin from the base Docker image, which produces different results than pipeline_v1.bin.

---

## Summary of Answers

| Question | Answer |
|----------|--------|
| Q1 | uv 0.9.5 |
| Q2 | sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e |
| Q3 | 0.533 |
| Q4 | 0.534 |
| Q5 | 121 MB |
| Q6 | 0.99 |

---

## Files Created

1. **05-deployment.ipynb** - Jupyter notebook with all solutions
2. **homework5_project/** - uv project directory with:
   - `pyproject.toml` - Project dependencies
   - `uv.lock` - Lock file with package hashes
   - `predict.py` - FastAPI service for pipeline_v1.bin
   - `predict_docker.py` - FastAPI service for pipeline_v2.bin
   - `Dockerfile` - Docker configuration
   - `test_q3.py`, `test_q4.py`, `test_q6.py` - Test scripts

---

## Commands to Run

### Start FastAPI Service (Q4)
```bash
cd homework5_project
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

### Build and Run Docker Container (Q6)
```bash
cd homework5_project
docker build -t homework5-model .
docker run -d -p 8000:8000 --name homework5-container homework5-model
```

### Stop Docker Container
```bash
docker stop homework5-container
docker rm homework5-container
```

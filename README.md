Here's a comprehensive `README.md` file for your project:

---

# Interpretable Predictions

## Overview

This project demonstrates how to build an interpretable machine learning system for predicting whether a user will be interested in an entity (e.g., clicking an advertisement). The system includes a **FastAPI** backend for API services, a **Streamlit** frontend for interactive visualizations, and **DiCE** for counterfactual explanations.

### Key Features:
1. Train a binary classifier to predict user interest.
2. Provide **LIME-based explanations** for predictions.
3. Generate **counterfactual explanations** for actionable insights.
4. Interactive visualization via **Streamlit**.

---

## Directory Structure

```
INTER_PRED
├── app
│   ├── main.py                # FastAPI backend
│   ├── streamlit_app.py       # Streamlit frontend
│   ├── preprocess.py          # Data preprocessing scripts
│   ├── model.py               # Model training and prediction logic
│   ├── counterfactuals.py     # Counterfactual generation logic
││   ├── Dockerfile             # Docker configuration
├── data
│   ├── ad_10000records.csv    # Dataset for training and predictions
├── notebooks
│   ├── Interpretable_Predictions.ipynb  # Development notebook
├── README.md                  # Project documentation
├── requirements.txt       # Python dependencies
```

---

## Setup Instructions

### Prerequisites
- **Python 3.10+**
- **Docker**

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/thanagno/interpretable-predictions.git
   cd interpretable-predictions
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend (FastAPI):
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. Run the frontend (Streamlit):
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. Access the application:
   - **FastAPI (Backend)**: [http://localhost:8000](http://localhost:8000)
   - **Streamlit (Frontend)**: [http://localhost:8501](http://localhost:8501)

---

## Using Docker

1. Build the Docker image:
   ```bash
   docker build -t interpretable-predictions .
   ```

2. Run the Docker container:
   ```bash
   docker run -d -p 8000:8000 -p 8501:8501 interpretable-predictions
   ```

3. Access the application:
   - **FastAPI (Backend)**: [http://localhost:8000](http://localhost:8000)
   - **Streamlit (Frontend)**: [http://localhost:8501](http://localhost:8501)

---

## Usage

1. **FastAPI**:
   - Test the backend with `/predict` and `/explain` endpoints.
   - Example request:
     ```bash
     curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"user_data": {...}, "entity_data": {...}}'
     ```

2. **Streamlit**:
   - Visualize model performance and explanations.
   - Generate counterfactual explanations for predictions.

---

## File Descriptions

- **`main.py`**:
  Implements FastAPI endpoints for predictions and explanations.
- **`streamlit_app.py`**:
  Provides an interactive dashboard for users to explore predictions and explanations.
- **`preprocess.py`**:
  Handles data preprocessing steps such as encoding and cleaning.
- **`model.py`**:
  Contains the logic for model training, saving, and loading.
- **`counterfactuals.py`**:
  Uses DiCE to generate counterfactuals.
- **`requirements.txt`**:
  Lists all dependencies for the project.

---

## Technologies Used

- **FastAPI** for the backend.
- **Streamlit** for interactive visualization.
- **LIME** for prediction explanations.
- **DiCE** for counterfactual generation.
- **Scikit-learn** for model training.
- **Docker** for containerization.

---

## Future Enhancements

- Extend the frontend with more detailed visualizations.
- Add support for real-time data ingestion.
- Implement advanced counterfactual techniques.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 
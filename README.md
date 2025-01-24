# Credit Scoring Model for Bati Bank

## Project Overview
The **Credit Scoring Model for Bati Bank** is designed to facilitate a "buy-now-pay-later" system by evaluating users' creditworthiness. The model classifies users as **high-risk** or **low-risk** based on their transaction data, enabling the bank to minimize default risks and enhance loan decision-making. The project encompasses data preprocessing, feature engineering, model development, API integration, and deployment.

---

## Features

### 1. Data Preprocessing
- **Handling missing values and outliers** to ensure robust analysis.
- **Encoding categorical variables** and normalizing numerical data for uniformity.

### 2. Feature Engineering
- **Aggregate features**: 
  - Total transaction amount
  - Average transaction value
  - Transaction count
  - Variability
- **Extracted features**: 
  - Transaction hour, month, and year
- **RFMS-based default estimator**: Utilizes Weight of Evidence (WoE) binning.

### 3. Model Development
- Trained models:
  - **Random Forest**
  - **Gradient Boosting Machines (GBM)**
- **Hyperparameter tuning** for improved performance and accuracy.

### 4. API Development
- **FastAPI** used to build a REST API for real-time predictions:
  - Endpoints to receive input data.
  - Returns classification results as high-risk or low-risk.

### 5. Deployment
- **Dockerized application** for seamless and portable deployment.
- **CI/CD pipeline** for automated testing and deployment to production.

### 6. Visualization Dashboard
- **Interactive dashboard** includes:
  - Fraud trends
  - Risk categories
  - Geographic data visualizations

---

## Installation

### Prerequisites
Ensure the following are installed on your system:

- Python 3.9 or higher
- Docker
- Git

### Clone the Repository
```bash
git clone https://github.com/Ofgeha-Gelana/CreditScoreAI.git
cd CreditScoreAI
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Running the API Locally
- Navigate to the project directory:
  ```bash
  cd CreditScoreAI
  ```
- Start the FastAPI server:
  ```bash
  uvicorn app.main:app --reload
  ```
- Access the API documentation at `http://127.0.0.1:8000/docs`.

### 2. Running the Dockerized Application
- Build the Docker image:
  ```bash
  docker build -t credit-score-api .
  ```
- Run the Docker container:
  ```bash
  docker run -p 8000:8000 credit-score-api
  ```
- Access the API at `http://127.0.0.1:8000`.

---

## CI/CD Pipeline
- **Automated Testing**:
  - Ensure all tests pass before deployment.
  ```bash
  pytest
  ```
- **Deployment**:
  - Continuous deployment via GitHub Actions.

---

## Visualization Dashboard
- Launch the dashboard:
  ```bash
  streamlit run dashboard/app.py
  ```
- View it in your browser at `http://localhost:8501`.

---

## Contributing
Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Open a pull request with a detailed description.

---


---

## Contact
For inquiries or support, please contact:
- **Author**: Ofgeha Gelana

---


# CodTech Intern — ML Projects Web App

This project bundles four small ML notebook demos into a single Flask web app. Each notebook's logic was extracted into `models/` modules and trained at app startup.

Files:
- `app.py` — Flask application entry point.
- `models` — backend notebook modules for each project
- `templates/` — HTML templates (Bootstrap 5).
- `static/style.css` — small custom CSS.
- `requirements.txt` — Python dependencies.

Quick start (Windows PowerShell):

```powershell
# create and activate virtualenv (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Notes:
- Models are trained in memory when the Flask app starts (no pickles used).
- The UI is intentionally simple; you can extend the forms and add validation.
# CodTechIntern

This repository contains a collection of machine learning mini-projects completed as part of the **CodTech Intern** program. Each notebook demonstrates the use of different ML techniques and libraries to solve practical problems such as classification, sentiment analysis, and recommendation systems.

---

## 📁 Project Structure

- **Customer Churn Prediction**  
  Predict whether a customer will discontinue a service based on account data, usage patterns, and tenure using classification models like Logistic Regression and Random Forest.

- **Breast Cancer Diagnosis**  
  A medical machine learning model that predicts whether a tumor is benign or malignant using Gradient Boosting and diagnostic feature analysis.
  
- **Sentiment Analysis**  
  Classifies text sentiment as positive or negative using Natural Language Processing with TF-IDF vectorization and Logistic Regression.
  
- **Stock Price Forecasting**  
  Predicts future stock closing prices based on recent historical data using regression and time-series modeling techniques.

---

## 🛠️ Technologies Used

- Python 3.x
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- NLP Tools 

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bukyajagadishnaik/CodTechIntern.git
cd CodTechIntern

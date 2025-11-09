# CodTech Intern — ML Projects Web App

This project bundles four small ML notebook demos into a single Flask web app. Each notebook's logic was extracted into `codtech_backend/` modules and trained at app startup.

Files:
- `app.py` — Flask application entry point.
- `codtech_backend/` — backend modules for each project (project1..project4).
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

- **Decision Tree Implementation.ipynb**  
  A notebook demonstrating how to build and visualize a decision tree classifier using scikit-learn.

- **Image Classification Model Using an Inbuilt Dataset.ipynb**  
  Uses built-in datasets from Keras or scikit-learn to build a CNN or DNN for image classification tasks.

- **Recommendation System Using Collaborative Filtering.ipynb**  
  A collaborative filtering-based recommendation system using user-item interaction matrices.

- **Sentiment Analysis Using TF-IDF and Logistic Regression.ipynb**  
  Implements a logistic regression model with TF-IDF features to classify text sentiment (positive/negative).

---

## 🛠️ Technologies Used

- Python 3.x
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- TensorFlow / Keras (for image classification)
- NLP Tools (TF-IDF from `sklearn.feature_extraction.text`)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bukyajagadishnaik/CodTechIntern.git
cd CodTechIntern

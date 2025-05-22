#  Fake News Detection System

A machine learning-powered web application that determines whether a **news headline (title)** is **fake** or **real**. The project uses **Logistic Regression** and **TF-IDF vectorization**, and is deployed using **FastAPI** for backend communication.

##  Features

- **Input**: Only the *title* of the news article
- **Preprocessing**: Lowercasing, punctuation removal, stopword filtering, and stemming
- **Model**: Logistic Regression (trained on the WELFake dataset)
- **Vectorization**: TF-IDF (Top 5000 terms)
- **Deployment**: FastAPI RESTful API
- **Output**: Classification label – "FAKE" or "REAL"

---

##  Project Structure

FAKE/
├── classifier2.pkl # Trained Logistic Regression model
├── vectorizer2.pkl # TF-IDF vectorizer (fitted on news titles)
├── main.py # FastAPI backend server
├── requirements.txt # Project dependencies
├── render.yaml # Render deployment configuration
└── README.md # Project documentation


---

## Sample Input 

{
  "title": "NASA confirms alien life discovered on Mars"
}

## Sample Output
{
  "prediction": "FAKE"
}


## How to Run Locally

1) Clone the repository
git clone https://github.com/pandyasmit-08/FAKE.git
cd FAKE

2) Install the dependencies
pip install -r requirements.txt

3) Start the FastAPI server
uvicorn main:app --reload

4) Open Swagger UI
Visit http://127.0.0.1:8000/docs in your browser to test the API using the interactive interface.


## Model Overview
Dataset	= WELFake (only titles used)
Model	= Logistic Regression
Vectorization =	TF-IDF (Top 5000 features)
Accuracy = ~90% on test set


## Deployment
1) Deployed using Render (cloud platform)
2) Backend accepts only news titles via a POST API
3) Ready to be integrated into any frontend (e.g., mobile app or web UI)


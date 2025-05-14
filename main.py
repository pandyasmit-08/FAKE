# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model and vectorizer
with open("classifier2.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer2.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI(title="Fake News Detection API")

class NewsItem(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Fake News Detection API is running."}

@app.post("/predict")
def predict(news: NewsItem):
    try:
        vectorized = vectorizer.transform([news.text])
        prediction = model.predict(vectorized)[0]
        result = "Real" if prediction == 1 else "Fake"
        return {"prediction": result}
    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

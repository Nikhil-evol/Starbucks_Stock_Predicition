from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from model_utils import load_model_and_scaler, prepare_input_for_inference, predict_n_days
import pandas as pd
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

COMPANIES = ["SBUX"]

model, scaler = load_model_and_scaler()
class StockData(BaseModel):
    recent_closes: list[float]

@app.get("/")
def form(request: Request, company: str = "SBUX"):
    return templates.TemplateResponse("index.html", {"request": request, "companies": COMPANIES, "company": company})


@app.post("/predict_form")
def predict_form(request: Request, company: str = Form("SBUX"), days: int = Form(7)):
    try:
        csv_path = os.path.join(os.path.dirname(__file__), f"{company}.csv")
        if not os.path.exists(csv_path):
            return templates.TemplateResponse("index.html", {"request": request, "error": f"CSV file not found at {csv_path}", "companies": COMPANIES, "company": company})

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if 'Close' not in df.columns:
            return templates.TemplateResponse("index.html", {"request": request, "error": "CSV does not contain 'Close' column.", "companies": COMPANIES, "company": company})

        closes = df['Close'].dropna().tolist()
        window_size = 1000
        if len(closes) < window_size:
            return templates.TemplateResponse("index.html", {"request": request, "error": f"Insufficient data in CSV: need at least {window_size} closes.", "companies": COMPANIES, "company": company})

        recent = closes[-window_size:]
        inv_preds = predict_n_days(model=model, scaler=scaler, recent_data=recent, window_size=window_size, n_days=days)
        preds_rounded = [f"{x:.2f}" for x in inv_preds]
        return templates.TemplateResponse("index.html", {"request": request, "predictions": preds_rounded, "companies": COMPANIES, "company": company})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e), "companies": COMPANIES, "company": company})


if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)

from fastapi import FastAPI
app=FastAPI()
@app.get("/")
def welcome():
   return{"mes":"Helloworld"}
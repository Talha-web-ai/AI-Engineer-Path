from fastapi import FastAPI
from app import routes

app = FastAPI(
    title="FastAPI + CNN Demo",
    description="Learning FastAPI step by step",
    version="1.0"
)

# include routes
app.include_router(routes.router)

@app.get("/")
def read_root():
    return {"message": "🚀 Hello World! FastAPI is running."}

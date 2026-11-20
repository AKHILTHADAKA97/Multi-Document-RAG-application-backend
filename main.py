"""Production and Development Entrypoint for Atlas Backend.

Compatible with:
- Render / Cloud Run / Heroku: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Alternative import paths:   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Direct execution:          `python main.py`
"""
import os
import uvicorn
from app.main import app, create_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)

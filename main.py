from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import threading
import webbrowser
import time

from webapp.api.appserver import router as api_router

app = FastAPI()
app.include_router(api_router)

templates = Jinja2Templates(directory="webapp")

app.mount("/static", StaticFiles(directory="webapp/static", html=True, check_dir=False))

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000/")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
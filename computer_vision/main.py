from fastapi import FastAPI
import uvicorn
from app.routers.detection import router as detection_router


app = FastAPI(
    title="Powerline Defect Detector",
    version="0.1.0",
)


# Подключаем роутер
app.include_router(detection_router)


# Локальный запуск через `python app/main.py`
if __name__ == "__main__":


    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

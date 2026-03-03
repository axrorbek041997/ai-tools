from fastapi import FastAPI

from modules import router

app = FastAPI()

app.include_router(router, prefix='/api')

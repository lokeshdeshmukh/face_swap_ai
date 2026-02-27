from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import cors_origins, ensure_data_dirs, settings
from app.core.container import container
from app.db.base import Base
from app.db.session import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_prefix)


@app.on_event("startup")
async def on_startup() -> None:
    ensure_data_dirs()
    Base.metadata.create_all(bind=engine)
    await container.queue_provider.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await container.queue_provider.stop()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

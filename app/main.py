from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.v2.upscale import router, thread_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting upscaling service...")
    yield

    thread_pool.shutdown(wait=True)
    print("Shutting down upscaling service...")

app = FastAPI(lifespan=lifespan)

app.include_router(router, prefix="/api/v2")

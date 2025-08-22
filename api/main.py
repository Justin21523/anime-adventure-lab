from fastapi import FastAPI
from api.middleware import setup_middleware
from api.routers import health, llm, rag, t2i, vlm, finetune, batch, admin

app = FastAPI(title="SagaForge API", version="0.1.0")
setup_middleware(app)

app.include_router(health.router)
app.include_router(llm.router)
app.include_router(rag.router)
app.include_router(t2i.router)
app.include_router(vlm.router)
app.include_router(finetune.router)
app.include_router(batch.router)
app.include_router(admin.router)

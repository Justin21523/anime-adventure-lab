from celery import Celery
import os
celery_app = Celery("sagaforge", broker=os.getenv("REDIS_URL","redis://localhost:6379/0"))
@celery_app.task
def ping(): return "pong"

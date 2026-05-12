import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Backend-side app: only sends tasks, never consumes results via Celery.
# Status is tracked via the database (polling /scans/{id}), not Celery result backend.
app = Celery("dental", broker=REDIS_URL)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,   # don't try to connect to result backend on send_task
)

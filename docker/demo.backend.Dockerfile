FROM python:3.10-slim

WORKDIR /app

COPY docker/requirements.demo.txt ./docker/requirements.demo.txt
RUN pip install --no-cache-dir -r docker/requirements.demo.txt

COPY webapp ./webapp

EXPOSE 8000
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]


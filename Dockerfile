FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Auto-setup if data/model are missing
RUN python -c "import os; os.makedirs('data',exist_ok=True); os.makedirs('model',exist_ok=True)"

EXPOSE 8000

CMD ["sh", "-c", \
     "[ -f model/route_model.pkl ] || python scripts/setup.py && \
      uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2"]

# 1 — Use a lightweight official Python image
FROM python:3.12-slim

# 2 — Work inside /app
WORKDIR /app

# 3 — Install Python dependencies first (for layer-caching)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4 — Copy the rest of your project
COPY . .

# 5 — Expose Streamlit’s default port
EXPOSE 8501

# 6 — Start Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.10-slim

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit

# Copy the rest of the application
COPY . .

# Expose the API port and Streamlit port
EXPOSE 8000
EXPOSE 8501

# Default command to run the API (can be overridden to run Streamlit)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

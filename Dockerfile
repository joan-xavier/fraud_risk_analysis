# Use official Streamlit image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first, then install (caches better)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all other project files (including .pkl files)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

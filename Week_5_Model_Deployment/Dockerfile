FROM python:3.8.12-slim

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy the application code
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Expose the application port
EXPOSE 9696

# Start the application
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

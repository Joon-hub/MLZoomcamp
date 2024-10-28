# Use the official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile ./
COPY Pipfile.lock ./
COPY model_C=1_splits=5.bin ./

# Install dependencies
RUN pipenv install --deploy --ignore-pipfile

# Copy the application code into the container
COPY predict.py ./

# Set the command to run the application
CMD ["pipenv", "run", "waitress-serve", "--host=0.0.0.0", "--port=5001", "predict:app"]

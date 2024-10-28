# Use a lightweight Python image as the base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install pipenv to manage dependencies
RUN pip install --no-cache-dir pipenv

# Copy the Pipfile and Pipfile.lock (if exists) to install dependencies
COPY Pipfile ./
COPY Pipfile.lock ./

# Install dependencies using pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the application code into the container
COPY predict.py ./

# Expose the application port
EXPOSE 5001

# Command to run the application using waitress
ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=5001", "predict:app"]

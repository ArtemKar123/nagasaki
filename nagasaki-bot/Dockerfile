FROM python:3.10-slim

ENV PORT 5001
ENV TG_API_KEY ""

# Set work directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

RUN apt-get update -y

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . /app

# Start the application
CMD ["python", "app.py"]
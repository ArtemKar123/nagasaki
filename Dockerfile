FROM python:3.10-slim

# Set environment variables
ENV HOST 0.0.0.0
ENV PORT 5001
ENV FLASK_APP /app/app.py

# Set work directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

RUN apt-get update -y && apt-get install -y gcc ffmpeg libsm6 libxext6

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . /app

# Expose the application's port
EXPOSE $PORT

# Start the application
CMD ["sh", "-c", "python -u $FLASK_APP --host=$HOST --port=$PORT"]
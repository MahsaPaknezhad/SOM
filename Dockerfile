# Use a base Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt /

# Install pip and packages from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Define the command to run the script
CMD ["python", "som.py"]

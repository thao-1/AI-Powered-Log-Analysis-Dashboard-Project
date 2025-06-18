# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY Requirements.txt /app/

# Install any needed packages specified in Requirements.txt
RUN pip install --no-cache-dir -r Requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Define environment variable
ENV MODULE_NAME app
ENV VARIABLE_NAME server

# Run app.py when the container launches
# The command should be gunicorn app:server -b 0.0.0.0:8050
# app refers to app.py, and server refers to the Dash app instance variable within app.py
# We need to ensure that app.py has a line like `server = app.server` (Dash convention)
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:server"]

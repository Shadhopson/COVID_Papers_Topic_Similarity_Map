# Start with the Python 3.8 base image
FROM python:3.8-slim-buster

# Sets the working directory for later commands
# When running the containing, you'll be in the directory specified here
WORKDIR /ui

# Install all our needed Python packages within the container
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy all the source code of the UI into the container
# Note: Rather than copying the code here, using a volume for easier code updates/testing
# COPY . .

# Launch the Bokeh App
CMD ["bokeh", "serve", "--show", "ui_app"]
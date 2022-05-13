# Start with base image
FROM jupyter/pyspark-notebook:8d32a5208ca1

USER root

# Install all our needed Python packages within the container
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# The following is necessary for PySpark to utilize the MySQL Driver
# The JDBC driver mentioned here was downloaded from the MySQL website manually
COPY ./mysql-connector-java_8.0.23-1ubuntu20.04_all.deb ./mysql-connector-java_8.0.23-1ubuntu20.04_all.deb
RUN dpkg -i mysql-connector-java_8.0.23-1ubuntu20.04_all.deb
RUN apt-get update

# Copy the source code and sample data into the container.
# This will allow the notebooks to be seen from the Jupyter UI
COPY ./*.ipynb ./
COPY ./sample_data ./sample_data/

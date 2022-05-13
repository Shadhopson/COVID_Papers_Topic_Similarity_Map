# Team 094 - DnD: Data and Depictions - README

## Note
It's recommended to view this file as markdown rather than text.

## Overview
Our application is split into 2 main components: 1) precomputation of data preparation and algorithms and 2) rendering in the UI for user exploration and interaction. Portions of this code have been executed at scale using infrastructure in AWS, while later portions, after much of the computation in AWS has completed, has been executed locally.

To execute the code base, you can run it at-scale in AWS, then pull the results down locally for loading to the UI. Alternatively, you can run everything locally using a series of Docker Containers with a smaller sample data set.

The below directions are split based on how you want to demo the code. There are some steps which are common between the full-scale and small-scale options. These steps should be executed regardless of the option selected. For the small-scale or full-scale demo, please follow the appropriate set of instructions.


## Team Members
* Scott Cardinal
* Shad Hopson
* Keegan Nesbitt
* Tina Nguyen
* Erhan Posluk
* Annmarie Thomas


## Demo Instructions
### Install Docker Locally
First, you need to be sure Docker Desktop is installed and running on your local machine. To do so, please follow the [instructions here](https://docs.docker.com/get-docker/).

### Docker Resource Configurations
Later steps in this code are rather CPU and memory-intensive. To run the code successfully, it's recommended to update your Docker Desktop settings to the following values.

Docker Desktop --> Settings --> Resources --> Advanced
```text
CPUs: 6
Memory: 10.00 GB
Swap: 1 GB
```

### MySQL Docker Container Setup
When running the code locally, you'll need to save and read data from a MySQL database. We can set one up locally using a Docker Container. However, to enable interaction between the MySQL Container and our other containers for running our application code, we'll need to do some setup.

Prior to running any of the following commands, open a terminal session and `cd` to the directory containing this README file and all the code.

First, we'll need to create 2 Docker Volumes for MySQL. This will allow the Docker Container to run the database, but any data it needs to persist will be stored outside of the container, allowing the container to be started/stopped without losing any state.
```bash
docker volume create cse6242_team094_mysql
docker volume create cse6242_team094_mysql_config
```

Next, we'll need to create a Docker Network. This will allow the other application code, also running within a Docker Container, to interact with the MySQL Database.
```bash
docker network create cse6242_team094_mysqlnet
```

Now we're ready to start the MySQL Database itself. The below command will download the mysql image from the Docker Registry, attach the 2 Docker Volumes we created earlier, and utilize the Docker Network we created. The mysql image is a little under ~600MB, so it may take some time to download.
```bash
# Mac, Linux
docker run --rm -d \
  -v "$(pwd)/cse6242_team094_mysql":/var/lib/mysql \
  -v cse6242_team094_mysql_config:/etc/mysql -p 3306:3306 \
  --network cse6242_team094_mysqlnet \
  --name cse6242_team094_mysqldb \
  -e MYSQL_ROOT_PASSWORD=p@ssw0rd1 \
  mysql

# Windows
# First ensure you have the needed drive shared in the Docker settings. See the below link for details.
# https://docs.docker.com/docker-for-windows/#file-sharing
docker run --rm -d -v "%cd%/cse6242_team094_mysql":/var/lib/mysql -v cse6242_team094_mysql_config:/etc/mysql -p 3306:3306 --network cse6242_team094_mysqlnet --name cse6242_team094_mysqldb -e MYSQL_ROOT_PASSWORD=p@ssw0rd1 mysql
```

Next, we need to create the MySQL database schema we'll utilize. To interact with the MySQL server, you can execute the below command. The password requested should match to the one utilized in the prior `docker run` command. Once logged in, you should be able to query against the database tables like you would in a typical MySQL session.
```bash
docker exec -ti cse6242_team094_mysqldb mysql -u root -p
```

Then, run the following command to create the needed database schema and select it for future commands.
```sql
CREATE DATABASE cse6242_team094;
```

To close out of the MySQL session, run the following:
```sql
exit;
```

Because of the usage of the Docker Volumes, even if you stop and restart this MySQL container, you'll still maintain any data you inserted into the database.

>**Note:** As you walk through the demo if you're running locally, you can view an example row of the contents from each table. To do so, you'll need to utilize the above `docker exec` command and enter the following, changing `TABLE_NAME` with an actual table name you want to view.
```sql
USE cse6242_team094;
SELECT * FROM TABLE_NAME LIMIT 1;
```


### Data Processing Option 1 - Local (Small-Scale) Demo Instructions
#### Build Docker Image
Our code is designed to run in Jupyter using a combination of PySpark and other libraries. We created a Docker Image with the necessary dependencies to help ensure the code can easily run locally.

First, you should change directories into the `data_processing` directory.
```bash
cd data_processing
```

Next, we'll build the Docker Image that includes Jupyter, PySpark, and other needed dependencies.  
Estimated Runtime: ~10-15 minutes  
Estimated Image Size: ~6 GB  
```bash
docker build --tag cse6242_team094_data_preproc .
```

#### Run Docker Image
Once the above Docker Container is built, launch it using the below command. This docker command creates a volume, which can be used for loading data to MySQL (see later instructions). It also exposes port 8888 so you can access Jupyter.
```bash
# Mac, Linux
docker run -it \
    -v "$(pwd)/aws_data":/home/jovyan/aws_data \
    -p 8888:8888 \
    --network cse6242_team094_mysqlnet \
    --name cse6242_team094_data_preproc_container \
    cse6242_team094_data_preproc

# Windows
# First ensure you have the needed drive shared in the Docker settings. See the below link for details.
# https://docs.docker.com/docker-for-windows/#file-sharing
docker run -it -v "%cd%/aws_data":/home/jovyan/aws_data -p 8888:8888 --network cse6242_team094_mysqlnet --name cse6242_team094_data_preproc_container cse6242_team094_data_preproc
```

After entering this command, you should see a localhost URL (127.0.0.1) displayed in your terminal, like what's shown below. Copy the displayed URL to a browser (we tested with Google Chrome). Be sure to take the URL your run prints, not what's shown below as the token will be unique for each execution.
```text
http://127.0.0.1:8888/?token=d3b82856dc547604ae9eec79540a4e4f88b52e254134956a
```

>**Note:** If you accidentally stop the Docker container, you'll need to delete the `cse6242_team094_data_preproc` Container from the Docker Dashboard and rerun the `docker run` command above from your terminal. Otherwise, you'll get an error saying the container name is already in use.

#### Code Execution
First, open the `01-PySpark_Lemmatization.ipynb` notebook in the Jupyter UI. If you're prompted for a Kernel, ensure `Python 3` is selected.

At the start of this notebook, you'll see a `run_mode` setting. Be sure this is set to `LOCAL_RUN_MODE`. After doing so, you can run the notebook end-to-end. The final output will be persisted into the MySQL Database running in the separate container we started earlier in this guide.

Next, open the `02a-BERTopic.ipynb` notebook in the Jupyter UI after closing/halting the first notebook. As with the first notebook, ensure the `run_mode` is set to `LOCAL_RUN_MODE`. This will retrieve the data from the first step from MySQL and write the outputs into a separate set of MySQL tables. Do the same with the `02b-LDA.ipynb` notebook.

With the topics generated, the next step is to cluster them. To run this step, within Jupyter open the `03-TopicClustering.ipynb` notebook. After running this code, you'll see two new tables generated, representing the edge lists of a topic graph and paper graph.

> **Note:** The `03-TopicClustering.ipynb` notebook is rather memory intensive, utilizing a little over 8-9GB at peak. Be sure you updated your Docker Desktop resource settings as mentioned at the start of this guide. You'll need to run this notebook twice, once for BERTopic and once for LDA. See the notebook for more details.

### Data Processing Option 2 - AWS (Full-Scale) Demo Instructions
The AWS demo leverages two services from the AWS suite: S3 and EMR. Start by visiting the AWS Management Console, which can be accessed through the following link(s):
- For full users: [AWS Management Console](https://aws.amazon.com/console/)
- For student accounts: [AWS Login Student Users](https://aws.amazon.com/education/awseducate/) 

#### Setup S3 Directories and Files
S3 will be used for storage, as both a read and write location of the data. These will include the following: 
- Read location for the raw csv data file of unprocessed text.
- Read location for a bash script to load the required packages to all the computing machines. 
- Write location for complete pre-processed text data, in the format of parquet files. 
To begin the S3 setup, navigate to the AWS S3 page, which can be found under *Storage* category in the AWS Management Console. Create a bucket in S3 by clicking the *Create Bucket* button (orange) on the right side of the page. 
> **Note:** AWS has requirements on bucket names that requires all names to be unique (similar to how usernames must be unique). A bucket name is provided in these instructions which was not taken at the time of writing these instructions, but if the bucket name is taken please select any unique name that works. 
For the bucket details, please select the following (default settings for the remaining items):
```text
Name: cse6242-team094-spring2021
Region: US East (N. Virginia)
```

Next, create two subfolders within the bucket called `read` and `write`. Upload the sample data csv files (sample file recommended to save time) and the bash files into the read folder. The folder structure of the two newly created folders should appear as follows:
```text
/read
--emr_bootstrap.sh
--metadata_sample_5000.csv
--metadata.csv (optional, large file)

/write
--empty (until files are written from PySpark)
```
> **Note:** The full dataset was not provided in csv format with the folders for this project (due to the already large size and the data being translated to a SQL database). If the full dataset is desired for the AWS preprocessing, please download the metadata.csv file from the [Kaggle Competition Website](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

#### Setup EMR and PySpark script
Navigate back to the AWS Console Management page and select EMR under the *Analytics* category. Select *Clusters* from the left-hand toolbar and select *Create Cluster*. At the top of the page, select *Go To Advanced Options* and select the following settings (if a setting isn't listed, select the default setting)

Step 1 - Software and Steps
```text
Release: emr-5.33.0
Software Config (check only the following): 
- Spark 2.4.7
- JupyterEnterpriseGateway 2.1.0
```
Step 2 - Hardware
```text
Cluster node and instances: 
- Master - m5.xlarge - 1 Instance Count
- Core - m5.xlarge - 4 Instance Count
- Task - m5.xlarge - 0 Instance Count
```
Step 3 - General Cluster Settings
```text
Cluster name: dnd-team094-cluster
Bootstrap actions:
- Add bootstrap action: Custom action
-- Configure and add: 
--- Name: Bootstrap procedure
--- Script location: Navigate to read folder and select bash script (emr_bootstrap.sh)
```
Step 4 - Security
```text
EMR role: EMR_DefaultRole
EC2 instance profile: EMR_EC2_DefaultRole
```
Select *Create Cluster*. Once the cluster is created and is in the "Ready" state, select *Notebooks* from the left-hand toolbar and click Create Notebook. Select the following settings for the Notebook:
```text
Notebook name: dnd-team094-notebook
Cluster: Choose an existing cluster - Select the cluster that was just created (dnd-team094-cluster)
Notebook location: cse6242-team094-spring2021
```
Create the notebook and once it reaches the "Ready" status, open it in Jupyter Lab. Once open, click the arrow with a line under in the left-hand bar to upload the `01-PySpark_Lemmatization.ipynb` file from the project download. Open the file (from the left side toolbar) and select PySpark for the kernel when prompted (or alternatively select the kernel in the top-right of the window). At the start of this notebook, you'll see a `run_mode` setting. Be sure this is set to `AWS_EMR_RUN_MODE`.

Now the PySpark can be run in sequence to transform the raw data to pre-processed text. The output will contain 3 subfolders in the write folder created earlier (in S3):
- abstract_parquet: A folder for processed abstract data.
- titles_parquet: A folder for title data.
- metadata_parquet: A folder for other metadata (authors, publishers, publication dates)

The final output will be persisted into the S3 bucket you created. These files will need to be downloaded from S3 and loaded into a local MySQL database for use with the UI. This is described in a later portion of this guide.

One thing to note during these instructions is the possibility of the original bucket name being taken. If that is the case, select a bucket name that isn't taken and adjust the steps above accordingly with the new bucket name. However, there is also a portion of the PySpark code that will need to be adjusted to accomodate the new read & write locations. The code in the following lines should be adjusted accordingly to the updated locations: 
```python
read_bucket = "s3://cse6242-team094-spring2021/read"
write_bucket = "s3://cse6242-team094-spring2021/write"
```

#### Setting Up An EC2 Instance
Now that we've run the NLP preprocessing code in AWS EMR, we can run the BERTopic and LDA modelling code on an AWS EC2 instance.

To set up an EC2 instance go to AWS, go to the EC2 section, and then select "Launch instances". From there select the Ubuntu Server 20.04 LTS (HVM), SSD Volume Type AMI and the m5.24xlarge instance type. Hit next until you get to "Configure Security Group" and make sure that you have type SSH going into Port Range 22. Hit Review and Launch and your instance is up. In order to give it a larger disk drive we'll need to replace the current 8 GB volume. To do so go back to your instances, and click on the name of the one you've just made. There in the "Networking" tab write down the availability zone, and from the "Storage" tab write down the root device name. Click on the volume ID then when you are in the volume tab with that volume selected hit "Actions" and then "Create Snapshot". This will take a few minutes to complete. You can check progeress in the snapshots section of the side directory. When it's ready you can then click "Create Volume". Here set the size to 100 GiB, set "Availability Zone" to the one you wrote down, and select the snapshot you just made. Click Create Volume and put in the root device name you noted earlier. This volume can now be added to your instance by stopping your instance, and then in the "Volumes" section removing the volume it has currently by selecting that volume and choosing "Detach Volume" from "Actions", and attaching the new larger volume by choosing that one, selecting "Attach Volume" from "Actions" and selecting the instance you made. You can now start your instance again. 

#### Connecting To Your EC2 Instance
How you connect to your EC2 Instance is dependent on your operating system, it's recommended that you use the official guide:
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html 
In our project we connect via ssh, and transfer files via WinSCP. Instructions for both can be found in the guide.

#### Setting Up Your Environment
Once you are in your EC2 instance's terminal, you can run the following commands:

``` bash
wget https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html
bash Anaconda3-2020.11-Linux-x86_64
```

This will download anaconda, we then need to put it into your path, so type 
```bash
vim .bashrc
```
and add this line near the top
``` 
export PATH=~/anaconda3/bin:$PATH
```
in vim you can use i to insert text, and then esc, followed by :wq to save and exit.

Type
```bash
source .bashrc
```
into the terminal and conda will be in your path. 

We can now begin downloading the necessary libraries by running the following lines one after another. This set up will work for running both models, if you only want to run LDA you do not need to install bertopic.
```bash
conda install pip
sudo apt update
sudo apt install build-essential
pip install bertopic
pip install pyarrow
sudo apt-get install zip unzip
```
with that your environment should be all set.

#### Running A Model
From here you can use WinSCP, or whatever other file transfer program you're using to move the parquet files and the python program for running the model over to your instance. Once you've done that just unzip the parquet files, and from there any parquet files in the abstract_parquet folder will be used by the models. To match our results remove all files except 0-9, and then you can run the python file of the model you would like to run. BERTopic takes around 3.5 hours to complete with 10 parquet files, and LDA takes about an hour.
```bash
unzip abstract_parquet.zip
python {Insert File Name}
```

You may run the `03-TopicClustering.ipynb` notebook at scale following the same set of steps as above for BERTopic and LDA.

> **Note:** In the above, you're running the Python file, not the notebook. You'll need to download a .py version from Jupyter if you choose this approach.


### User Interface Instructions
#### Load Data to MySQL
> **Note:** This step is only necessary for loading the full-scale data from AWS and can otherwise be skipped. If using the demo options in the code, everything has already been loaded into MySQL.

First, download the Parquet files from AWS S3, keeping the files grouped into the same directories as they were saved into within the S3 Bucket. Place each of these files into the respective folders present with in the `/data_processing/aws_data` directories. The folders mentioned in `aws_data` correspond to the output directories mentioned in the code.

After you place the Parquet files from S3 into these directories, it should look something like what's pictured here for the `01-abstracts` directory:
```text
/data_processing/aws_data/01-abstracts
-- _placeholder
-- part-00000-9fa7ae3b-ad42-4fd1-b13d-28ab11c41e3a-c000.snappy.parquet
-- part-00001-9fa7ae3b-ad42-4fd1-b13d-28ab11c41e3a-c000.snappy.parquet
-- ...
-- part-00025-9fa7ae3b-ad42-4fd1-b13d-28ab11c41e3a-c000.snappy.parquet
```

From the `02a-BERTopic.ipynb` and `02b-LDA.ipynb` notebooks, load the output CSVs into the corresponding directories within `/data_processing/aws_data` as well. As with the `_placeholder` file present in the above Parquet file directories, the `.placeholder` file can remain here.

The full-scale output of `03-TopicClustering.ipynb` should be placed into the directories prefixed with `03-`. There are directories present for the graphs representing both the BERTopic and LDA approaches.

Using the `cse6242_team094_data_preproc_container` Docker container defined above and the Jupyter UI it launches. Within Jupyter, navigate to the `UI_Data_Load.ipynb` notebook. This notebook will be utilized to load the data from downloaded Parquet files into the MySQL database. No changes are required within this notebook, and each cell can be run as-is.

#### Build Docker Image
First, ensure your terminal's current directory is directly within the `user_interface` folder.  
Estimated Runtime: ~1 minute  
Estimated Image Size: ~500 MB  
```bash
docker build --tag cse6242_team094_ui .
```

#### Run Docker Image & Launch User Interface
Once the UI Docker Container is built, launch it using the below command.
```bash
# Mac, Linux
docker run -it \
    -v "$(pwd)/ui_app":/ui/ui_app \
    -p 5006:5006 \
    --network cse6242_team094_mysqlnet \
    --name cse6242_team094_ui_container \
    cse6242_team094_ui

# Windows
# First ensure you have the needed drive shared in the Docker settings. See the below link for details.
# https://docs.docker.com/docker-for-windows/#file-sharing
docker run -it -v "%cd%/ui_app":/ui/ui_app  -p 5006:5006 --network cse6242_team094_mysqlnet --name cse6242_team094_ui_container cse6242_team094_ui
```

After running the above command, you should see a URL printed in the terminal like what's shown in the following. Copy this URL to the browser (we tested with Google Chrome) to view the UI application.
```text
Bokeh app running at: http://localhost:5006/ui_app
```


## Cleanup Instructions
This section outlines steps you can take after running the demo to clean up anything you setup locally to free space on your local computer.

To stop the MySQL Docker Container:
```bash
docker container stop cse6242_team094_mysqldb
```

After changing directory to the same directory as this Readme, remove the folder used by the MySQL Docker Volume:
```bash
# Mac, Linux
rm -rf cse6242_team094_mysql

# Windows
del -rf cse6242_team094_mysql
```

To remove the container you created:
```bash
docker container rm preproc_container
docker container rm cse6242_team094_ui_container
```

To remove any built images that are no longer required, you can execute the following.
```bash
docker image rm cse6242_team094_data_preproc
docker image rm cse6242_team094_ui
```

To remove the created volumes:
```bash
docker volume rm cse6242_team094_mysql
docker volume rm cse6242_team094_mysql_config
```

To remove the network we created:
```bash
docker network rm cse6242_team094_mysqlnet
```

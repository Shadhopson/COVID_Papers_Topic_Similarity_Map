#!/bin/bash
sudo yum -y install gcc
sudo yum -y install python3-devel
sudo pip-3.7 install -U \
    pyahocorasick==1.4.1 \
    contractions
sudo mkdir /usr/lib/nltk_data
sudo python3.7 -m nltk.downloader -d /usr/lib/nltk_data averaged_perceptron_tagger
sudo python3.7 -m nltk.downloader -d /usr/lib/nltk_data stopwords
sudo python3.7 -m nltk.downloader -d /usr/lib/nltk_data wordnet
sudo python3.7 -m nltk.downloader -d /usr/lib/nltk_data punkt
sudo python3.7 -m nltk.downloader -d /usr/lib/nltk_data omw
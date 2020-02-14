#!/usr/bin/env bash

# max 90 gb required
# upload these to the server
# .kaggle/, bam_dataset_download/*.py, model/ PBN-convert-multi*

pip install kaggle;

apt-get -y install libsm6 libxrender-dev libxrender1 libfontconfig1
pip install opencv-python
python3 -m pip install opencv-contrib-python

pip install tqdm;
pip install imageio;
pip install comet_ml;
pip install matplotlib
apt-get -y install python3-tk;


# Fetch prepickled bam dataset data from google drive. This contains urls to be crawled
wget https://gist.githubusercontent.com/schmidtdominik/4d520346c6e5e528f51b332bb7bb8788/raw/0a332b580098c84003370fcdab2afc575252e3ff/dl_from_gdrive.py;
mkdir bam_dataset_download
echo Downloading pickled bam url data
python3 dl_from_gdrive.py 1iCTIDKGZhWHaOEiGSKiQo9AV5B1wFa_O ./bam_dataset_download/bam.pickle
rm dl_from_gdrive.py


kaggle competitions download painter-by-numbers -f train.zip;
unzip train.zip;
rm train.zip

kaggle competitions download painter-by-numbers -f test.zip;
unzip test.zip;
rm test.zip

mv ./test/* ./train/
rmdir test


# While/After this:
# crawl as much BAM data as you want # Warning partial downloads cannot be resumed if following commands have already been
# executed since the "for f in ... mv" move command doesn't overwrite but instead renames files
# cd bam_dataset_download; python3 bam_crawler.py; cd

# After all this
# Merge painter by numbers kaggle data in ./train with bam_dataset_download crawled files:
# for f in ./bam_dataset_download/downloaded/*; do mv --backup=t "$f" ./train/; done
# https://unix.stackexchange.com/questions/371375/mv-add-number-to-file-name-if-the-target-exists
# crop and scale everything
#cd; python3 PBN-convert-multiproc.py


#ls -1 | wc -l

# pip install jupyter;
# jupyter lab --ip=127.0.0.1 --port=8080 --allow-root
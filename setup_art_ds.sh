echo
# min 100GB gb required
# upload these to the server
# .kaggle/, bam_dataset_download/*.py, model/ PBN-convert-multi*

# alternatively, download celebA
# python3 dl_from_gdrive.py 1e0ESqLn9ViaD3pFwVvUHL-6Y5I5yffgP ./celeba.zip
# mkdir celeba
# mv celeba.zip celeba/celeba.zip
# cd celeba && unzip celeba.zip

mkdir bam_dataset_download
echo Downloading prepickled bam url data
python3 dl_from_gdrive.py 1iCTIDKGZhWHaOEiGSKiQo9AV5B1wFa_O ./data/bam_dataset_download/bam.pickle

kaggle competitions download painter-by-numbers -f train.zip;
unzip train.zip;
rm train.zip

kaggle competitions download painter-by-numbers -f test.zip;
unzip test.zip;
rm test.zip

mv ./test/* ./train/
rmdir test
mv train data/train

# While/After this:
# crawl as much BAM data as you want # Warning partial downloads cannot be resumed if following commands have already been
# executed since the "for f in ... mv" move command doesn't overwrite but instead renames files
# cd data/bam_dataset_download; python3 bam_crawler.py; cd

# After all this
# Merge painter by numbers kaggle data in ./train with bam_dataset_download crawled files:
# for f in ./data/bam_dataset_download/downloaded/*; do mv --backup=t "$f" ./data/train/; done
# https://unix.stackexchange.com/questions/371375/mv-add-number-to-file-name-if-the-target-exists
# crop and scale everything
#cd; python3 performance_crop.py

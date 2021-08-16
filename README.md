# deepinsight-iqa

#### Execute the below command to begin Training 
```
python deepinsight_iqa/cli.py train -mdiqa -b/root/deepinsight-iqa/ -cconfs/diqa_conf.json -fcombine.csv -i/root/dataset --train_model=all --load_model=all
```

#### For prediction run the below command
```
python deepinsight_iqa/cli.py predict -mdiqa -b/root/deepinsight-iqa/ -cconfs/diqa_conf.json -i/root/dataset/live/jp2k/img1.bmp 
```
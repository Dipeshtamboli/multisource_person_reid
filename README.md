<h1 align="center"> Person re-identification utilizing multiple source domains </h1>

Here, we are using [Layumi's Person re-ID](https://github.com/layumi/Person_reID_baseline_pytorch) code for extracting features and to create baseline model. So citing it as it is and also using it's readme file with suitable chamges.

## What is domain shift?

### Separately trained and extracted the features
Following is the TSNE plot of the features extractred for the different identities of Market1501 and DutkeMTMC dataset.
First we trained our model on Market1501 dataset and extracted query identities of the same Market1501.
Then we trained our model in DukeMTMC dataset and extracted query identities of the same DukeMTMC.   
Following is the combined TSNE of plot of both the dataset's 512-d features where we can see the different geometric spacial location. In 512-d dimension, both set of the features are far apart and thus domain shift is there.

![](https://github.com/Dipeshtamboli/multisource_person_reid/blob/master/code/tsne/market_duke.png)

### Cross trained and extracted the features
#### Trained on DukeMTMC dataset and query extracted from Market1501 dataset
![](https://github.com/Dipeshtamboli/multisource_person_reid/blob/master/code/tsne/m_d_dm_tsne.png)
Here, our model is trained on DukeMTMC dataset ans the query images are from Market1501 dataset. In this TSNE, 

> Blue-> Market1501 features from the model trained on Market1501   
> Orange-> DukeMTMC features from the model trained on DukeMTMC and  
> Green-> **Market1501 features from the model trained on DukeMTMC**

We can see that the cross-Market1501(Market features extracted from the model trained on DukeMTMC dataset) features are mixed with DukeMTMC features partially. Although we used the same extractor(model trained on Duke) for DukeMTMC and Market1501 still their features are not overlapping and have some **shift** between them.

#### Trained on Market1501 dataset and query extracted from DukeMTMC dataset
![](https://github.com/Dipeshtamboli/multisource_person_reid/blob/master/code/tsne/m_d_md_tsne.png)
Here, our model is trained on Market1501 dataset ans the query images are from DukeMTMC dataset. In this TSNE, 

> Blue-> Market1501 features from the model trained on Market1501   
> Orange-> DukeMTMC features from the model trained on DukeMTMC and   
> Green-> **DukeMTMC features from the model trained on Market1501**   

Similarly what we saw above,here,
We can see that the cross-Duke(Duke features extracted from the model trained on Market1501 dataset) features are mixed with Market1501 features partially. Although we used the same extractor(model trained on Market1501) for Market and duke still their features are not overlapping and have some **shift** is there between them.

#### Both cross-examples are included
> Blue-> Market1501 features from the model trained on Market1501   
> Orange-> DukeMTMC features from the model trained on DukeMTMC and   
> Green-> **Market1501 features from the model trained on DukeMTMC**   
> Red-> **DukeMTMC features from the model trained on Market1501**   

![](https://github.com/Dipeshtamboli/multisource_person_reid/blob/master/code/tsne/all_m_d_dm_md.png)

Like mentioned above, you can checked the cross-feauture mixing here.

### Query-result illustration for domain shift
Where all the results are good for the models trained and tested on the same dataset, not a single result is correct for the cross-models(trained on one and tested on other) due to domain shift.
#### Model tested on Market1501 and trained on DukeMTMC and Market respectively
![](https://github.com/Dipeshtamboli/multisource_person_reid/blob/master/code/reid_results/mdm.png)
#### Model tested on DukeMTMC and trained on Duke and Market respectively
![](https://github.com/Dipeshtamboli/multisource_person_reid/blob/master/code/reid_results/dmd.png)

==================================================================================

### Prepare Data Folder (`python prepare.py`)
Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)
You may notice that the downloaded folder is organized as:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```
Open and edit the script `prepare.py` in the editor. Change the fifth line in `prepare.py` to your download path, such as `\home\zzd\Download\Market`. Run this script in the terminal.
```bash
python prepare.py
```
We create a subfolder called `pytorch` under the download folder. 
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
```

In every subdir, such as `pytorch/train/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. 

### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


### Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

### A simple visualization (`python demo.py`)
To visualize the result, 
```
python demo.py --query_index 777
```
`--query_index ` which query you want to test. You may select a number in the range of `0 ~ 3367`.


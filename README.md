**Paper_Replication_Reimaging_of_Price_Trend**
***
HKUST MAFM course project
This project is one of my class assignments completed at HKUST M.Sc. in Financial Mathematics (MAFM) program. This is a PyTorch implementation of 
[**(Re-)Imag(in)ing Price Trends**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587).
Check the pdf report for detailed description.

## Config Setting
```Batch
./config/xxx.yam;
```
Since I basically use the Kaggle as training platform. So you may need to rewrite the img_path and model_path parameters. Or you can also use kaggle platform since I have
upload my train-test dataset on kaggle platform([data source](https://www.kaggle.com/datasets/aliawran/dataset-of-reimaging-price-trend)).

## Quick Start
Train Mode:
```Batch
!python main.py "xxx.yml" -t
```
Inference Mode:
```Batch
!python main.py "xxx.yml" -i
```
As for the back test result, you shoul check the *test.tpynb* and *backtesting.py* file.
You can also use the kaggle as training and predicting platform, my project files are also public in kaggle([project file](https://www.kaggle.com/datasets/aliawran/project-file10)), which
 is basically same as this repo. Please feel free give this repo a star if you find it useful and please raise issue if you find sth confused.

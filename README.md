# CH4Net 🛰️

A fast, simple model for detection of methane plumes. For more information see our paper at https://egusphere.copernicus.org/preprints/2023/egusphere-2023-563/

## Data
Data are available from zenodo at https://zenodo.org/deposit/8267966. 

## Training
To train CH4Net with all sentinel-2 bands as predictors
```
python3 train.py 12 {output directory}
```
To train CH4Net with only bands 11 and 12 as predictors
```
python3 train.py 2 {output directory}
```

## Reference
```
@article{vaughan2023ch4net,
  title={CH4Net: a deep learning model for monitoring methane super-emitters with Sentinel-2 imagery},
  author={Vaughan, Anna and Mateo-Garc{\'\i}a, Gonzalo and G{\'o}mez-Chova, Luis and R{\uu}{\v{z}}i{\v{c}}ka, V{\'\i}t and Guanter, Luis and Irakulis-Loitxate, Itziar},
  journal={EGUsphere},
  volume={2023},
  pages={1--17},
  year={2023},
  publisher={Copernicus Publications G{\"o}ttingen, Germany}
}
```

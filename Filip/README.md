## Usage of Classifer

Command to generate dataset folder, with parametars from classification/config.py

```python
class confuguration:
    kernel_size = 80
    db_path = './../dataset_csv_format/'
    classes = ['N', 'L', 'R', 'A', 'V']
    limit_class_N = 50000 
```

```bash
cd ./classification
python multyclassifier.py
```



To test model it is needed to adjust directory of model from file test_model.py:

```python
loaded_model = joblib.load('models/svm.sav')
```

To run test:

```bash
python test_model.py
```


## Usage of GAN

Command to generate dataset folder, with parametars from gan/config.py

```python
class confuguration:
    kernel_size = 80 
    db_path = './../dataset_csv_format/'
    classes = ['N', 'L', 'R', 'A', 'V']
    limit_class_N = 50000 
```

```bash
cd ./gan
python load_dataset.py
python gan.py
```

## TODO

- [ ] Create requirment libery file
- [ ] Extend configuration file 
- [ ] Refactor code
- [ ] Debug memory Leak problem
- [ ] GAN with diffrent number of layers
- [ ] GAN with wavelet transformation
- [ ] Generate new dataset and compare the results with the original dataset





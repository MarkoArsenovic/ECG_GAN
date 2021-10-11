## Customize the dataset

Change parametrs from configuration.py:

```python
class confuguration:
    kernel_size = 80
    db_path = './dataset_csv_format/'
    classes = ['N', 'L', 'R', 'A', 'V']
    limit_class_N = 50000 
```
Extract database from MIT-BIH csv format:

```bash
python load_dataset.py
```

## Usage of Classifer

The commands for training classification models:

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


## Usage of DCGAN

Set a parametars for learning, from the file gan/DCGAN/gan.py:

```python
dataset_path = "./dataset/"
batch_size = 128
epochs = 50
learning_rate_D = 0.00018
learning_rate_G = 0.0002
noise = 32
```

Execute training:


```bash
cd ./gan/DCGAN
python gan.py
```

## Usage of WGAN



Set a parametars for learning, from the file gan/DCGAN/gan.py:

```python
dataset_path = "./dataset/"
batch_size = 128
epochs = 50
learning_rate_D = 0.00018
learning_rate_G = 0.0002
noise = 32
```

Import dataset and execute training with:


```bash
cd ./gan/WGAN
python wgan.py
```

As for wavelet transformation use:


```bash
python wgan_wl.py
```

## Usage of PCGAN

Start with training:

```bash
cd ./gan/PCGAN
python train256 --batch_size=256 --channel=128 --init_step=2 --lrD=0.0008 --lrG=0.001 --n_critic=1 --path='./../../dataset/' --pixel_norm=False --tanh=True --total_iter=10000 --trial_name='experiment-1' --z_dim=125
```

## Results

**Classificacion models are trained on original dataset:**

Linear SVM

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |26219| 98  | 54  |69   | 101
(L) | 125 |2704 | 0   |1    | 5
(R) | 227 |1    |1983 |297  | 0
(A) | 552 |2    |5    |1470 | 5
(V) | 557 |56   |28   |49   | 1116


Linear SVM with SDG

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |26374| 26  | 22  | 66  | 53
(L) | 293 |2536 |  0  |  1  | 5
(R) | 429 |  0  |1891 | 188 | 0
(A) | 697 |  1  |  1  |1329 | 6
(V) | 749 | 28  | 13  | 50  | 966


Logistic regression

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |26269| 87  | 41  | 54  | 90
(L) | 123 |2694 |  1  |  3  | 14
(R) | 274 |  0  |1955 | 279 | 0
(A) | 582 |  3  |  6  |1437 | 6
(V) | 571 | 58  | 33  | 52  | 1092


                            |   Accuracy  |   Recall    |   Precision   |   F1 score
--------------------------  | ----------- | ----------- | ------------- | ------------- 
Linear SVM                  |  **0.937**  |  **0.814**  |     0.907     |  **0.853**   
Linear SVM with SDG         |    0.929    |    0.770    |     0.915     |    0.823     
Logistic regression         |    0.936    |    0.806    |   **0.936**   |    0.806     

**Classificacion models are trained on original dataset, with synthetic 2048 samples for each class, using PCGAN:**

Linear SVM

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |-147 |-34  |-7   | 53  | 67
(L) | 4   |-8   | 1   | 0   | 3
(R) | 11  | 0   | 23  |-34  | 0
(A) |-9   | 2   | 2   | 5   | 0
(V) |-68  | 22  | 1   | 16  | 31


Linear SVM with SDG

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |-378 | 259 | 12  | 80  | 51
(L) |-16  | 22  | 0   | 0   | 4
(R) |-49  | 0   |-14  | 63  | 0
(A) |-48  | 1   |-1   | 50  | 1
(V) |-195 | 47  |-9   | 84  | 125


Logistic regression

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |-175 | 64  | 2   | 22  | 87
(L) | 9   |-5   | 2   |-1   |-5
(R) |-4   | 0   | 50  |-46  | 0
(A) |-13  | 0   |-1   | 15  |-1
(V) |-70  | 21  |-2   | 5   | 46


                            |      Accuracy    |      Recall     |     Precision    |   F1 score
--------------------------  | ---------------- | --------------- | ---------------- | ------------- 
Linear SVM                  |  0.936 (-0.001)  |  0.816 (+0.004) |  0.893 (-0.014)  |  0.851 (-0.002)
Linear SVM with SDG         |  0.922 (-0.007)  |  0.787 (+0.011) |  0.876 (-0.039)  |  0.814 (-0.009)
Logistic regression         |  0.934 (-0.002)  |  0.815 (+0.009) |  0.896 (-0.040)  |  0.806 (+0.044)

**Classificacion models are trained on original dataset, with synthetic 2048 samples for each class except N, using PCGAN:**

Linear SVM

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |-245 | 52  |-7   | 92  | 108
(L) | 7   |-12  | 1   | 1   | 3
(R) | 31  |-1   | 28  |-58  | 0
(A) |-35  | 3   | 3   | 27  | 2
(V) |-86  | 27  |-5   | 20  | 44

Linear SVM with SDG

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |-767 | 234 | 18  | 116 | 399
(L) |-26  | 13  | 2   | 0   | 11
(R) |-71  | 0   | 138 |-68  | 1
(A) |-152 | 4   | 5   | 140  | 4
(V) |-334 | 27  | 13  | 22  | 324

Logistic regression

    | (N) | (L) | (R) | (A) | (V)
--- | --- | --- | --- | --- | --- 
(N) |-378 | 115 | 0   | 87  | 168
(L) | 16  |-17  | 3   | 2   |-4
(R) |-6   | 0   | 92  |-86  | 0
(A) |-55  | 0   |-1   | 58  |-2
(V) |-118 | 23  | 0   | 76  | 71


                            |      Accuracy    |      Recall     |     Precision    |   F1 score
--------------------------  | ---------------- | --------------- | ---------------- | ------------- 
Linear SVM                  |  0.933 (-0.0039) |  0.821 (+0.0076)|  0.885 (-0.0211) |  0.849 (-0.0034)
Linear SVM with SDG         |  0.923 (-0.0055) |  0.820 (+0.0509)|  0.848 (-0.0661) |  0.832 (+0.0094)
Logistic regression         |  0.931 (-0.0046) |  0.822 (+0.0169)|  0.880 (-0.0553) |  0.848 (+0.0835)



## TODO

- [ ] Create requirment libery file
- [ ] Extend configuration file 
- [ ] Refactor code
- [ ] Debug memory Leak problem
- [ ] GAN with diffrent number of layers
- [ ] GAN with wavelet transformation
- [ ] Generate new dataset and compare the results with the original dataset





# Cat vs Dog Classifier

CNN that classifies images as cat or dog, built in TensorFlow/Keras. I trained it with 5 different optimizers to see which one actually performs best on this dataset.

## What it does

Images go in at 128x128, get run through 3 conv+pooling blocks (32, 64, 128 filters), then flattened into a dense layer and a single sigmoid output for the cat/dog decision. Trained for 5 epochs per optimizer, binary cross-entropy loss.

## Results
**Optimizer full names:**
- SGD – Stochastic Gradient Descent
- Adam – Adaptive Moment Estimation
- Adagrad – Adaptive Gradient Algorithm
- RMSprop – Root Mean Square Propagation
- Nadam – Nesterov-accelerated Adaptive Moment Estimation

Tried SGD, Adam, Adagrad, RMSprop, and Nadam under the same setup:

| Optimizer | Accuracy | Loss | Time (s) |
|---|---|---|---|
| Nadam | 84.7% | 0.441 | 366 |
| Adam | 83.1% | 0.417 | 357 |
| RMSprop | 82.9% | 0.401 | 344 |
| SGD | 72.7% | 0.546 | 391 |
| Adagrad | 62.2% | 0.643 | 343 |

Nadam came out on top for accuracy, RMSprop had the lowest loss. SGD and Adagrad just didn't keep up in 5 epochs — probably need a lot more training time to catch up, if they ever would.

## Dataset

Using the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle. 

1. Download the dataset from the link above
2. Drop it into `data/train/` and `data/test/`, with `cats/` and `dogs/` subfolders in each
3. Update the paths at the top of the notebook if yours don't match

## Running it

Needs TensorFlow, NumPy, Pandas, Matplotlib, and scikit-learn. Once the dataset's in place, just open the notebook and run through the cells top to bottom — each optimizer trains and logs its results at the end.

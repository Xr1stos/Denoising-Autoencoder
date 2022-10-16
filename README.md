# Denoising-Autoencoder
An implementation of a denoising autoencoder for missing value imputation

## Autoencoders

Autoencoders are Neural Networks which try to reconstruct their input.
The network consists of two parts. The encoder, that tries to compress the
input into a new representation and the decoder, that tries to reconstruct
the original input from that representation (Goodfellow et al., 2016). Autoencoders can’t copy perfectly the input, as the representation layer is usually
smaller in size compared to the input. As a result, the network discovers
useful properties of the data and learns the most salient features in order to
minimize the reconstruction error. This is the main reason that Autoencoders
are commonly used for feature selection and dimensionality reduction.
The Autoencoder must be able to generalize and not just learn the input
data. Towards this goal, an approach is to slightly corrupt the input data but
maintain the uncorrupted data as the network’s target. This way, the model
can’t simple learn the input but it is forced to learn a mapping between the
input and a manifold. A manifold is an area in space where the input data![ae](https://user-images.githubusercontent.com/17834602/196054894-7e5872d9-d702-47aa-b8b7-04620169f500.png)

are concentrated (Goodfellow et al., 2016). So, if this manifold describes the
original data, the added noise is removed. Such a network is known as a
Denoising Autoencoder.


Figure 1: An Autoencoder network visualization. The hidden layer is the encoded input
representation. Then, the decoder tries to reconstruct the input from the hidden layer (Jordan, 2018).


Figure 2: A Denoising Autoencoder network visualization. The corrupted data are used as input and
the reconstructed representation is compared against the original uncorrupted data (Jordan, 2018).

## Denoising Autoencoder implementation
In our data, there are missing values, which can be considered as noise.
So what if an Autoencoder could learn the most salient features and then
could reconstruct not only the input data but also the missing values? So
the Denoising Autoencoder tries to undo the corruption done by the missing
values. In this case we are not concerned for the hidden layer but for the
reconstructed input. The steps of this process given a dataset with null values
are the following:

### Training phase
1. Create a mask containing missing values’ positions.
2. Fill the missing values with the mean of each variable.
3. Create pseudo missing values that don’t overlap with the original ones.
With this technique we have corrupted values as input but also the real
ones so that the Autoencoder can undo the corruption.
4. Augment the dataset by creating and concatenating permutations of the
original dataset with different pseudo missing values each time.
5. Fill the pseudo missing values with a dummy value that is out of the
range of values of all variables. It can be a negative or relatively large
value or even zero if it is not present in any variable. This value indicates
’missingness’ and tries to make the model focus on these cases, as the
reconstruction error for them will be high.
6. Train the Autoencoder using the dummy filled augmented dataset and
calculate the loss between the reconstructed dataset and the dataset
with the pseudo missing values having their known original values.

By filling the original missing values with mean (dummy value is used
for the pseudo missing values), all the dataset can be used and by creating
pseudo missing values the Autoencoder can be trained on existing correct
data. As a result, the Denoising Autoencoder tries to undo the corruption
done by the dummy values, so it is important that in the prediction phase the
same dummy value is used.

### Prediction phase
1. Fill the original missing values with the dummy value used during the
training.
2. Use the learned model to reconstruct the dummy filled original dataset
and get the predictions.

### System parameters

Below is a description of the most important parameters of the system.

* Pseudo missing values rate. Depends on the original percentage of missing values. Approximately 20% seems to be enough.
* Augmentation permutations. Depends on the dataset and the original
percentage of missing values. There seems to be a positive effect when
concatenating up to 30 permutations of the original dataset with different pseudo missing values.
* Dummy value. The corrupted values the Autoencoder tries to undo. It
must be a distinct value not contained in the set of values of the variables. Large values can have a negative effect on the training process.
* Network layers and nodes. Depend on the dataset but augmentation
tends to smooth results. Few layers, between 1 to 3, might result in
lower error during the training process, but will have greater error at the
prediction of the test set. These layers, also depending on the number
of available features, tend to result in a model with very few parameters,
which doesn’t help the network to learn and generalize. It is advised to
use 5 or more layers.
* Validation set. Used for validation during training. It was set to 30% of
the dataset.
* Loss function. Mean squared error tends to perform better than mean
absolute error.
* Optimizer. The optimizer used was Adam with a learning rate of 0.0005.
* Batch size. Batch size seems to affect the performance. Sizes used were
between 8 and 64. Small datasets tend perform better with smaller batch
size. In addition training with a smaller batch size tends to give better
prediction results.
* Early stopping with best epoch parameters was used.

The advantages of the above method are that it will perform better with
bigger datasets as it is a Neural Network and the **model can be saved and used
for future prediction in contrast to most imputation techniques which don’t
produce a model**. On the other hand parametrization can be time consuming

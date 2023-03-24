# Hyperbolic Autoencoder for Recommendations in JAX

Repository contains JAX implementation of AE model, which incorporates input information via embedding into Hyperbolic space, providing better accuracy when applied to collaborative filtering recommender systems.

The idea is taken from [Hyperbolic autoencoders for recommender systems](https://dl.acm.org/doi/10.1145/3383313.3412219) by E.Frolov et al. and [HyperbolicRecommenders](https://github.com/evfro/HyperbolicRecommenders) repo.

## Table of Contents

- [Requirements](#req)
- [Data](#data)
- [Code](#code)

## Requirements
In order to launch scripts, you need to install latest JAX version.
Also, our code is based on [Rieoptax](https://github.com/SaitejaUtpala/rieoptax) library, which provides JAX implementation of functions optimization on Riemanian Manifolds. 

## Data
All training and test data (basically MovieLens dataset) used for trainig AE model can be found in `data` folder.

## Code
Ipynb notebooks can be found in root directory.




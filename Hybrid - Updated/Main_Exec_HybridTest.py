# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Content
ContentKNN = ContentKNNAlgorithm()

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

#Combine them
Hybrid = HybridAlgorithm([SVDPlusPlus, ContentKNN], [0.5, 0.5])

evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")
evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(True)


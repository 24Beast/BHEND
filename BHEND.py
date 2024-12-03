# Importing Libraries
import torch
import numpy as np
from .utils import BiasPredictor, BiasFinder

# Main Class
class BHEND:
    def __init__(self, bias_finder_args : dict, bias_predictor_args : dict) -> None:
        self.initBiasFinder(bias_finder_args)
        self.initPredictor(bias_predictor_args)
        
    def initPredictor(self, bias_predictor_args: dict) -> None:
        # Add any preprocessing for input dict
        self.predictor = BiasPredictor(bias_predictor_args)
        # Add conditions for loading existing Predictor
        self.trainedPredictor = False
    
    def initBiasFinder(self, bias_finder_args) -> None:
        # Add any preprocessing for input dict
        self.finder = BiasFinder(bias_finder_args)
        # Add conditions for loading existing Finder
        self.trainedFinder = False
    
    def getSamples(self, X, y) -> dict:
        if(self.trainedFinder):
            labels = self.finder.label(X,y)
        else:
            self.finder.train(X,y)
            labels = self.finder.label(X,y)
        return labels
    
    def predictLabels(self, X) -> dict:
        if not(self.trainedPredictor):
            raise AssertionError("Predictor must be trained before labels can be generated!")
        label_preds = self.predictor.predict(X)
        return label_preds
    
    def predictTestDistribution(self, X_test) -> dict:
        if not(self.trainedPredictor):
            raise AssertionError("Predictor must be trained before test labels can be generated!")
        labels = self.predictLabels(X_test)
        num_items = len(labels)
        num_conflicting = sum([label for label in labels.values()])
        return {"Aligned": num_items - num_conflicting, "Conflicting": num_conflicting}
    
    def trainProcess(self, X, y) -> tuple:
        labels = self.getSamples(X, y)
        self.predictor.train(X, labels)
        self.trainedPredictor = True
        label_preds = self.predictor.predict(X)
        return labels, label_preds


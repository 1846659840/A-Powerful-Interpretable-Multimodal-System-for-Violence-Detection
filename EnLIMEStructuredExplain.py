#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# lime_interpreter.py

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class EnLimeInterpreter:
    def __init__(self, model, train_data,test_data=None,local_model='EnLIME-xgboost',num_samples=1000, kernel_width=0.25, weights_method='LIME'):
        """
        Initialize the LIME interpreter.

        Parameters:
        - model: the black box model to interpret.
        - train_data: the training data for the model.
        - num_samples: the number of perturbed samples to generate.
        - kernel_width: the width of the kernel function.
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.local_model = local_model
        self.weights_method = weights_method
        
        # Determine the types of features
        self.categorical_features = train_data.select_dtypes(include=['object', 'category']).columns
        self.numerical_features = train_data.select_dtypes(exclude=['object', 'category']).columns

    def generate_perturbations(self, instance):
        """
        Generate perturbations around the instance.

        Parameters:
        - instance: the instance around which to generate perturbations.

        Returns:
        - a pandas DataFrame of perturbations.
        """
        # Initialize DataFrame for perturbations
        perturbations = pd.DataFrame(columns=instance.columns, index=np.arange(self.num_samples))

        # Generate perturbations for categorical features
        for feature in self.categorical_features:
            # Calculate the probabilities of each category in the training data
            probs = self.train_data[feature].value_counts(normalize=True)
            # Generate random samples based on these probabilities
            perturbations[feature] = np.random.choice(probs.index, size=self.num_samples, p=probs.values)

        # Generate perturbations for numerical features
        for feature in self.numerical_features:
            perturbations[feature] = np.random.normal(0, 1, size=self.num_samples) * self.train_data[feature].std() + self.train_data[feature].mean()

        return perturbations

    def interpret(self, instance, weights_method='LIME', local_model=None):
        """
        Interpret the model's prediction at the instance.

        Parameters:
        - instance: the instance to interpret.

        Returns:
        - a pandas Series of feature importances.
        """
        if local_model is None:
            local_model = self.local_model
        
        
        instance = pd.DataFrame([instance], columns=self.train_data.columns)
        perturbations = self.generate_perturbations(instance)
        
        if weights_method == 'LIME':
            distances = pairwise_distances(perturbations, instance, metric='cosine').ravel()
            weights = np.sqrt(np.exp(-(distances**2) / self.kernel_width**2))
        elif weights_method == 'Kernel-SHAP':
            counts = perturbations.sum(axis=1)
            M = self.train_data.shape[1]
            weights = [(M - 1) / (comb(M, z) * z * (M - z)) for z in counts]
        else:
            
            raise ValueError('Invalid weights method.')
        
        predictions = self.model.predict(perturbations)
        if self.local_model == 'LIME-linear' or self.local_model == 'Kernel-SHAP-linear':
            local_model = LinearRegression()
            local_model.fit(perturbations, predictions, sample_weight=weights)
            importances = pd.Series(local_model.coef_, index=instance.columns)
        elif self.local_model == 'LIME-decision_tree' or self.local_model == 'Kernel-SHAP-decision_tree':
            local_model = DecisionTreeRegressor()
            local_model.fit(perturbations, predictions, sample_weight=weights)
            importances = pd.Series(local_model.feature_importances_, index=instance.columns)
        elif self.local_model == 'EnLIME-random_forest' or self.local_model == 'Kernel-SHAP-random_forest':
            local_model = RandomForestRegressor()
            local_model.fit(perturbations, predictions, sample_weight=weights)
            importances = pd.Series(local_model.feature_importances_, index=instance.columns)
        elif self.local_model == 'EnLIME-xgboost' or self.local_model == 'Kernel-SHAP-xgboost':
            local_model = XGBRegressor()
            local_model.fit(perturbations, predictions, sample_weight=weights)
            importances = pd.Series(local_model.get_booster().get_score(importance_type='weight'), index=instance.columns)
        else:
            raise ValueError('Invalid local model type.')

        return importances
    
    def plot_importance(self, importances, plot_title, global_interpret=False, instance=None):
        plt.figure(figsize=(10, 5))
        plt.title(plot_title)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        
        if global_interpret:  # If it's global interpretation, don't add feature values
            colors = ['blue' for _ in importances.values]
        else:  # If it's local interpretation, add feature values to feature names
            if instance is not None:
                importances.index = [f'{feature} = {value}' for feature, value in zip(importances.index, instance.values.flatten())]
            colors = ['red' if i < 0 else 'blue' for i in importances.values]

        importances.plot(kind='barh', color=colors)
        plt.show()
        
    def global_interpret(self, subset='train', local_model='LIME-linear', weights_method='LIME'):
  
    # Determine the subset of data to use
        if subset == 'train':
            data = self.train_data
        elif subset == 'test':
            data = self.test_data
        elif subset == 'all':
            data = pd.concat([self.train_data, self.test_data], ignore_index=True)
        else:
            raise ValueError('Invalid subset type.')

    # Initialize the importances series
        importances = pd.Series(np.zeros(self.train_data.shape[1]), index=self.train_data.columns)

    # Calculate local importances for each instance in the data
        for i, row in data.iterrows():
            instance_importance = self.interpret(row,weights_method,local_model)
            for feature in importances.index:
                if feature in instance_importance:
                    importances[feature] += abs(instance_importance[feature]) if local_model.startswith('LIME') else instance_importance[feature]
        return importances

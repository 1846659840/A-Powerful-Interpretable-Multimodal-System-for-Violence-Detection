
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import cv2
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
from scipy.special import comb


class EnLIMEExplain:
    def __init__(self, model, kernel_width=0.25, num_samples=10, perturb_ratio=[0.3, 0.7],regressor_type='random_forest',weight_type='lime', perturbation_type='fix'):
        self.model = model
        self.kernel_width = kernel_width
        self.num_samples = num_samples
        self.perturb_ratio = perturb_ratio
        self.regressor_type = regressor_type
        self.weight_type = weight_type
        self.perturbation_type = perturbation_type
        np.random.seed(222)
        warnings.filterwarnings('ignore')

    def read_and_preprocess_img(self, path):
        Xi = skimage.io.imread(path)
        Xi = skimage.transform.resize(Xi, (299,299))
        Xi = (Xi - 0.5)*2
        return Xi

    def predict(self, img): 
        preds = self.model.predict(img[np.newaxis,:,:,:], verbose=0)
        top_pred_classes = preds[0].argsort()[-6:][::-1]
        return preds, top_pred_classes

    def generate_superpixels(self, img):
        superpixels = skimage.segmentation.quickshift(img, kernel_size=4, max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        return superpixels, num_superpixels

    def perturb_image(self, img, perturbation, segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1 
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image*mask[:,:,np.newaxis]
        return perturbed_image

    def generate_perturbations(self, img, num_superpixels):
        if self.perturbation_type == 'fix':
            perturbations = []
            for _ in range(self.num_samples):
                 perturbations.append(np.random.choice(2, num_superpixels, p=self.perturb_ratio))
        elif self.perturbation_type == 'ran_fix':
            perturbations = np.random.binomial(1, 0.5, size=(self.num_samples, num_superpixels))
        else:
            raise ValueError("Invalid perturbation_type. Choose between 'fix' and 'ran_fix'.")
        return perturbations


    def explain_instance(self, path):
        Xi = self.read_and_preprocess_img(path)
        preds, top_pred_classes = self.predict(Xi)
        self.superpixels, self.num_superpixels = self.generate_superpixels(Xi)
        perturbations = self.generate_perturbations(Xi, self.num_superpixels)

        predictions = []
        for pert in perturbations:
            perturbed_img = self.perturb_image(Xi, pert,self.superpixels)
            pred = self.model.predict(perturbed_img[np.newaxis,:,:,:], verbose=0)
            predictions.append(pred)

        predictions = np.array(predictions)
        original_image = np.ones(self.num_superpixels)[np.newaxis,:]
        if self.weight_type == 'lime':
            distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
            weights = np.sqrt(np.exp(-(distances**2)/self.kernel_width**2))
        elif self.weight_type == 'kernel_shap':
            df = pd.DataFrame(perturbations)
            counts = df.sum(axis=1).to_numpy()
            M = self.num_superpixels
            weights = []
            for z in counts:
                weight = (M - 1) / (comb(M, z) * z * (M - z))
                weights.append(weight)
        else:
            raise ValueError("Invalid weight_type. Choose between 'lime' and 'kernel_shap'.")
        class_to_explain = top_pred_classes[0]
        if self.regressor_type == 'linear':
            regressor = LinearRegression()
        elif self.regressor_type == 'decision_tree':
            regressor = DecisionTreeRegressor()
        elif self.regressor_type == 'xgboost':
            regressor = XGBRegressor()
        else:
            regressor = RandomForestRegressor()
        regressor.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
        y_pred = regressor.predict(perturbations)

    # 根据模型类型获取重要性或系数
        if self.regressor_type == 'linear':
            self.importances = regressor.coef_[0]
        else:
            self.importances = regressor.feature_importances_

        return self.importances

    def visualize_top_importances(self, path, num_top_importances=10):
        Xi = self.read_and_preprocess_img(path)
        superpixels = self.superpixels
        importances = self.importances

  
        top_importances_indices = np.argsort(importances)[-num_top_importances:]

    # 创建一个遮罩，只显示前n个最重要的 superpixel
        mask = np.zeros(superpixels.shape)
        for active in top_importances_indices:
            mask[superpixels == active] = 1

    # 将原图像与遮罩合并，生成结果图像
        top_importance_img = copy.deepcopy(Xi)
        top_importance_img = top_importance_img * mask[:,:,np.newaxis]

    # 显示结果图像
        plt.imshow(top_importance_img)
        plt.show()

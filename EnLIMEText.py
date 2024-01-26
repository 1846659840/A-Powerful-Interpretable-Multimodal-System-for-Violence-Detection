import jieba
import unicodedata
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import xgboost as xgb
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def remove_punctuation_and_digits(text):
    # 定义一个正则表达式模式，用于匹配标点符号、特殊字符和数字
    pattern = r'[,.!?"\'，。！？“”‘’\d;:()（）【】<>《》【】「」『』【】@#￥%……&*]'
    # 使用正则表达式模式去除标点符号、特殊字符和数字
    text = re.sub(pattern, '', text)
    return text

def clean_text(text):
    # 去除标点符号、特殊字符和数字
    text_no_punct_and_digits = remove_punctuation_and_digits(text)
    # 使用结巴分词
    words = jieba.cut(text_no_punct_and_digits)
    return ' '.join(words)


def perturb_sentence(sentence, num_perturbations):
    words = sentence.split()
    perturbed_sentences = []
    for _ in range(num_perturbations):
        perturbed_words = words.copy()
        num_words_to_remove = random.randint(1, len(words)//2)
        indices_to_remove = random.sample(range(len(words)), num_words_to_remove)
        for index in sorted(indices_to_remove, reverse=True):
            del perturbed_words[index]
        perturbed_sentences.append(' '.join(perturbed_words))
    return perturbed_sentences

vectorizer = CountVectorizer(preprocessor=clean_text)

plt.rcParams["font.family"] = "Microsoft YaHei"
def plot_word_importances_Map(word_importances, original_sentence, model_name, offset=-0.15):
    # 提取原句中的单词和重要性值
    original_words, original_importance_values = zip(*word_importances[model_name])
    
    # 根据重要性值的范围选择红色渐变色
    cmap = plt.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=min(original_importance_values), vmax=max(original_importance_values))
    
    # 绘制原句
    plt.figure(figsize=(8, 5))
    plt.bar(original_words, original_importance_values, color=cmap(norm(original_importance_values)))
    plt.title(f"{model_name} 单词重要性", fontsize=16)  # 在标题中包含模型名称
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # 显示颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=12)
    
    # 显示原句（向下移动原句）
    plt.text(0.9, offset, original_sentence, ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    
def process_sentence(sentence, vectorizer, model, X_train, y_train, num_perturbations=1000):
    # 使用结巴进行分词
    sentence = ' '.join(jieba.cut(sentence))
    
    if not any(word in vectorizer.vocabulary_ for word in sentence.split()):
        raise ValueError('句子中没有任何单词在词汇表中.')
    # 生成扰动
    perturbed_sentences = perturb_sentence(sentence, num_perturbations)
   
    # 通过模型得到预测值
    perturbed_sentences_tfidf = vectorizer.transform(perturbed_sentences)
    svm_classifier = SVC(probability=True)
    svm_classifier.fit(X_train, y_train)
    predictions_proba = svm_classifier.predict_proba(perturbed_sentences_tfidf)
    predictions_proba_class_1 = predictions_proba[:, 0]
    
    # 计算权重    
    original_sentence_tfidf = vectorizer.transform([sentence])
    
    cosine_similarities = cosine_similarity(original_sentence_tfidf, perturbed_sentences_tfidf)
    kernel_width = 0.25
    rbf_similarities = np.exp(-0.5 * (1 - cosine_similarities) ** 2 / kernel_width**2)
    rbf_similarities_array = np.array(rbf_similarities[0])
    
    # 训练四个模型
    sentence_tokens = sentence.split()
    attributes_matrix = np.zeros((len(perturbed_sentences), len(sentence_tokens)))
    for i, perturbed_sentence in enumerate(perturbed_sentences):
        perturbed_sentence_tokens = perturbed_sentence.split()
        for j, original_token in enumerate(sentence_tokens):
            if original_token in perturbed_sentence_tokens:
                attributes_matrix[i, j] = 1

    models = [
        ("LIME", LinearRegression()),
        ("EnLIME-Random Forest", RandomForestRegressor()),
        ("LIME-Decision Tree", tree.DecisionTreeRegressor(max_depth=5)),
        ("EnLIME-XGBoost", xgb.XGBRegressor())
    ]
    importances = OrderedDict()
    for model_name, model in models:
        model.fit(attributes_matrix, predictions_proba_class_1, sample_weight=rbf_similarities_array)
        
        # 输出属性重要性
        if model_name == "LIME":
            importances[model_name] = list(zip(sentence_tokens, model.coef_))
        else:
            importances[model_name] = list(zip(sentence_tokens, model.feature_importances_))
    
    # 返回所有模型的重要性
    return importances











 


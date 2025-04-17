# xai_utils.py
# Utilities for explainable AI (SHAP/LIME)
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

def get_shap_explanation(model, data, background=None):
    explainer = shap.TreeExplainer(model, background if background is not None else data)
    shap_values = explainer.shap_values(data)
    return shap_values

def get_lime_explanation(model, data, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(data),
        feature_names=feature_names,
        mode="classification"
    )
    explanations = []
    for i in range(len(data)):
        exp = explainer.explain_instance(data.iloc[i], model.predict_proba, num_features=5)
        explanations.append(exp.as_list())
    return explanations

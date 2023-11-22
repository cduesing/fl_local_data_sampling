import numpy as np
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from copy import deepcopy
import model_utils
import models
import load_data
import strategies
from copy import deepcopy
from collections import Counter
import scikitplot as skplt
import matplotlib.pyplot as plt
#import shap
import pandas as pd
import seaborn as sns
from math import log2
from k_means_constrained import KMeansConstrained


# trains local models and compares them 
def run_local_baselines(config, central_model=None, filename=None, log_per_round=False):

    x_train_clients = config["clients_feature_dict"]
    y_train_clients = config["clients_label_dict"]

    count = Counter(config["y_test"])
    true_dict = {}
    for x in list(range(config["num_classes"])):
        if x not in count:
            true_dict[x] = 0
        else:
            true_dict[x] = count[x] / len(config["y_test"])
    t_percentage_count = [(i, true_dict[i]) for i in true_dict]

    local_performances, central_performances, local_metrics, central_metrics = [],[], [], []
    
    for j in range(config["num_clients"]):
        
        f1s = []
        best_model = None
        
        local_model = models.get_model_by_name(config)
        is_bert = config["model_name"].lower() == "bert"

        X_train, X_test, y_train, y_test = train_test_split(x_train_clients[j],
                                                            y_train_clients[j],
                                                            test_size = config["test_set_fraction"]
                                                           )
        count = Counter(y_train_clients[j])
        pred_dict = {}
        for x in list(range(config["num_classes"])):
            if x not in count:
                pred_dict[x] = 0
            else:
                pred_dict[x] = count[x] / len(y_train_clients[j])
        p_percentage_count = [(i, pred_dict[i]) for i in pred_dict]

        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test)
        
        for i in range(config["rounds"] * config["local_epochs"]*config["local_multiplier"]):
            local_model = model_utils.perform_training(local_model, X_train, y_train, config["batch_size"], 1, config["device"], is_bert=is_bert)
            
            predictions, probabilities = model_utils.perform_inference(local_model, X_test, config["batch_size"], config["device"], is_bert=is_bert)
            if config["num_classes"] > 1:
                _, _, f1, _ = precision_recall_fscore_support(y_test.numpy(), predictions, average=config["evaluation_averaging"])
            else: 
                f1 = mean_absolute_error(y_test.numpy(), probabilities)
            
            f1s.append(f1)

            if best_model is None:
                best_model = deepcopy(local_model)
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1: 
                best_model = deepcopy(local_model)
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] <= 1: 
                best_model = deepcopy(local_model)

        del local_model
        
        if best_model is not None:
            l_acc, l_pre, l_rec, l_f1, l_eval = evaluate_minority(best_model, X_test.numpy(), y_test.numpy(), config, filename, title="Local Data "+str(j) +" - Local Model")
            local_performances.extend(l_eval)
        else: 
            l_acc, l_pre, l_rec, l_f1, l_eval = 0,0,0,0,[]

        c_acc, c_f1 = 0,0
        if central_model is not None:
            c_acc, c_pre, c_rec, c_f1, c_eval = evaluate_minority(central_model, X_test.numpy(), y_test.numpy(), config, filename, title="Local Data "+str(j) +" - Central Model")
            central_performances.extend(c_eval)

        local_metrics.append((l_acc, l_pre, l_rec, l_f1))
        central_metrics.append((c_acc, c_pre, c_rec, c_f1))

        #_ = evaluate_minority(best_model, config["X_test"], config["y_test"], config, filename, title="Central Data - Local Model "+str(j))
        
    return local_performances, central_performances, local_metrics, central_metrics

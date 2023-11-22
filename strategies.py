import models
import model_utils
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error
import numpy as np
import sys


# used for inherence of different methods
class FedBlueprint:
    
    # init the learning strategy
    def __init__(self, config):
        self.config = config
        self.central_model = models.get_model_by_name(config)
        self.is_bert = config["model_name"].lower() == "bert"
    
    # runs training and evaluation procedure on provided data and config
    def run(self, config, filename=None, log_per_round=False, return_f1s=False):
        
        best_central_model = None
        
        writer = open(filename, 'w') if filename is not None else sys.stdout
        accuracies, precisions, recalls, f1s, all_distributions = [],[],[],[],[]
        
        for i in tqdm(range(config["rounds"])):
            self.train_round()
            acc, pre, rec, f1, all_predicitions = self.evaluate()

            if best_central_model is None:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] == 1:
                best_central_model = deepcopy(self.central_model.cpu())

            accuracies.append(acc)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            all_distributions.append(np.unique(all_predicitions, return_counts=True))
            
            if log_per_round:
                model_utils.write_metrices(writer, "Round "+str(i), acc, pre, rec, f1, np.unique(all_predicitions, return_counts=True), is_classification=self.config["num_classes"]>1)
        
        idx = np.argmax(f1s) if config["num_classes"] > 1 else np.argmin(f1s)
        model_utils.write_metrices(writer, "Best performance at round: "+str(idx), accuracies[idx], precisions[idx], recalls[idx], f1s[idx], all_distributions[idx], is_classification=self.config["num_classes"]>1)
        
        if filename is not None:
            writer.close()
        
        if return_f1s:
            return best_central_model, f1s

        return best_central_model

    # runs training and evaluation procedure on provided data and config
    def run_till_threshold(self, config, filename=None, log_per_round=False, return_f1s=False,f1_threshold=0.7):
        
        best_central_model = None
        
        writer = open(filename, 'w') if filename is not None else sys.stdout
        accuracies, precisions, recalls, f1s, all_distributions = [],[],[],[],[]
        
        num_till_f1 = None
        for i in tqdm(range(config["rounds"])):
            self.train_round()
            acc, pre, rec, f1, all_predicitions = self.evaluate()

            if best_central_model is None:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] == 1:
                best_central_model = deepcopy(self.central_model.cpu())

            accuracies.append(acc)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            all_distributions.append(np.unique(all_predicitions, return_counts=True))
            
            if log_per_round:
                model_utils.write_metrices(writer, "Round "+str(i), acc, pre, rec, f1, np.unique(all_predicitions, return_counts=True), is_classification=self.config["num_classes"]>1)

            if f1 >= f1_threshold:
                num_till_f1 = i+1
                break
        print("Rounds till convergence:", num_till_f1)

        idx = np.argmax(f1s) if config["num_classes"] > 1 else np.argmin(f1s)
        model_utils.write_metrices(writer, "Best performance at round: "+str(idx), accuracies[idx], precisions[idx], recalls[idx], f1s[idx], all_distributions[idx], is_classification=self.config["num_classes"]>1)
        
        if filename is not None:
            writer.close()
        
        if return_f1s:
            return best_central_model, f1s

        return best_central_model
    
    # computes accuracy, precision, recall and f1
    # takes test samples and labels
    # returns metrices
    def evaluate(self):

        x_test = torch.tensor(self.config["X_test"]).float()
        y_test = torch.tensor(self.config["y_test"])

        if self.config["num_classes"] > 1:    
            all_predicitions, _ = model_utils.perform_inference(self.central_model, x_test, self.config["batch_size"], self.config["device"], is_bert=self.is_bert)
            acc = np.sum(np.array(all_predicitions) == y_test.numpy()) / len(y_test)
            pre, rec, f1, _ = precision_recall_fscore_support(y_test, all_predicitions, average=self.config["evaluation_averaging"])
            return acc, pre, rec, f1, all_predicitions
        else: 
            _, all_predictions = model_utils.perform_inference(self.central_model, x_test, self.config["batch_size"], self.config["device"], is_bert=self.is_bert)
            mae = mean_absolute_error(y_test, all_predictions)
            mse = mean_squared_error(y_test, all_predictions, squared=True)
            rmse = mean_squared_error(y_test, all_predictions, squared=False)
            return _, mae, mse, rmse, all_predictions
        


# average model weights by number of samples for training
# with 'weighted=False', the model weights are averaged without being weighted by the client sample counter
class FedAvg(FedBlueprint):
    
    # aggregates the central model using weighted average from local models
    # performs a single learning round
    def train_round(self):
        local_models = []
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        for i in range(self.config["num_clients"]):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"], is_bert=self.is_bert)

            local_models.append(local_model)
            del local_model
        
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, y_train_clients, weighted=self.config["weighted"])
        
        
# takes a string as input
# returns a learning w.r.t. the provided name
# checks if all required arguments are provided
def get_strategy_by_name(config):
    if config["strategy_name"].lower() == "fedavg":
        print("FedAvg ignores these parameters: 'stepsize', 'reset_per_round'")
        return FedAvg(config)
    else:
        raise ValueError("strategy " + config["strategy_name"] + " has not been configured yet")
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.cluster import KMeans
import sys
import statistics
import torch
import torchvision
import torchvision.transforms as transforms
import random
import skews


# loads dataset from files and sets config parameters accordingly
def load_raw_data(config):
    
    if config["dataset_name"].lower() == "covtype":
        
        config["num_features"] = 54
        config["num_classes"] = 2

        if config["model_name"] == "auto":
            config["model_name"] = "ann"
        
        X,y = [],[]
        with open("./data/CovType/covtype.libsvm.binary.scale") as file:
            for line in file:
                s = line.rstrip()
                s = s.split(" ")
                y.append(int(s[0])-1) 
                xi = [0.] * config["num_features"]
                for e in s[1:]:
                    e = e.split(":")
                    i = int(e[0])
                    f = float(e[1])
                    xi[i-1] = f
                X.append(xi)
    
    elif config["dataset_name"].lower() == "hand":

        config["num_features"] = 15
        config["num_classes"] = 5

        if config["model_name"] == "auto":
            config["model_name"] = "ann"
        
        df = pd.read_csv("./data/hand_postures/allUsers.lcl.csv")

        y = list(df["Class"].to_numpy().astype(int))
        y = [x-1 for x in y]
        df = df.drop(["Class"], axis=1)
        X = list(df.to_numpy().astype(float))

    elif config["dataset_name"].lower() == "diabetes_insulin":

        config["num_features"] = 37
        config["num_classes"] = 2

        if config["model_name"] == "auto":
            config["model_name"] = "ann"
        
        df = pd.read_csv("./data/diabetes/diabetes.csv")

        y = list(df["change"].to_numpy().astype(int))
        df = df.drop(["change"], axis=1)
        X = list(df.to_numpy().astype(float))
        
    elif config["dataset_name"].lower() == "mnist":

        config["num_features"] = (1, 320)
        config["num_classes"] = 10

        if config["model_name"] == "auto":
            config["model_name"] = "cnn"

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        X = trainset.data[:,None,:,:].tolist()
        y = trainset.targets.tolist()

    elif config["dataset_name"].lower() == "cifar":

        config["num_features"] = (3, 500)
        config["num_classes"] = 10

        if config["model_name"] == "auto":
            config["model_name"] = "cnn"

        X,y = [],[]

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                                  shuffle=True, num_workers=2)
        for data in trainloader:
            inputs, labels = data
            X.extend(inputs.tolist())
            y.extend(labels.tolist())
        
    else:
        raise ValueError("dataset " + config["dataset_name"] + " has not been configured yet")
        
    config["X_raw"] = X
    config["y_raw"] = y

    return config


# apply skews and distribution on row data
def distribute_skewed_data(config):
    
    # train/shared/test splitting
    shared_frac = config["shared_set_fraction"]
    test_frac = config["test_set_fraction"]

    is_regression = not config["num_classes"] > 1
    num_classes = config["num_classes"] if not is_regression else config["num_quantiles"]
    
    X_train, X_shared_and_test, y_train, y_shared_and_test = train_test_split(config["X_raw"], 
                                                                              config["y_raw"], 
                                                                              test_size = test_frac + shared_frac
                                                                             )
    X_test, X_shared, y_test, y_shared = train_test_split(X_shared_and_test, 
                                                          y_shared_and_test, 
                                                          test_size = shared_frac / (shared_frac + test_frac)
                                                         )
    del config["X_raw"]
    del config["y_raw"]

    # apply either label or quantity skew
    if config["label_skew"] is None or config["label_skew"] == "homogeneous":
        sample_to_client_assignment = skews.apply_homogenous_data_distribution(X_train, y_train, config["num_clients"], num_classes)
    elif config["label_skew"] == "label_distribution" and config["label_alpha"] is not None:
        sample_to_client_assignment = skews.apply_label_distribution_skew(float(config["label_alpha"]), X_train, y_train, config["num_clients"], num_classes, is_regression)
    elif config["label_skew"] == "label_quantity" and config["label_n"] is not None:
        sample_to_client_assignment = skews.apply_label_quantity_skew(int(config["label_n"]), X_train, y_train, config["num_clients"], num_classes, is_regression)
    elif config["label_skew"] == "quantity" and config["label_alpha"] is not None:
        sample_to_client_assignment = skews.apply_quantity_skew(float(config["label_alpha"]), X_train, y_train, config["num_clients"], num_classes)
    elif config["label_skew"] == "clustering" and config["purity"] is not None:
        sample_to_client_assignment = skews.apply_clustering_skew(config["purity"], X_train, y_train, config["num_clients"], num_classes)
    else: 
        raise ValueError("label/quantity skew " + config["label_skew"] + " is not defined yet")
    
    # apply attribute skew
    if config["attribute_skew"] is None:
        clients_feature_dict, clients_label_dict = skews.apply_no_attribute_skew(X_train, y_train, sample_to_client_assignment)
    elif config["attribute_skew"] == "noise" and config["attribute_alpha"] is not None:
        clients_feature_dict, clients_label_dict = skews.apply_attribute_noise_skew(float(config["attribute_alpha"]), X_train, y_train, sample_to_client_assignment)
    elif config["attribute_skew"] == "availability" and config["attribute_alpha"] is not None:
        clients_feature_dict, clients_label_dict = skews.apply_attribute_availability_skew(float(config["attribute_alpha"]), X_train, y_train, sample_to_client_assignment)
    else: 
        raise ValueError("attribute skew " + config["attribute_skew"] + " is not defined yet")
    
    config["X_train"] = X_train
    config["y_train"] = y_train
    config["X_test"] = X_test
    config["y_test"] = y_test
    config["X_shared"] = X_shared
    config["y_shared"] = y_shared
    config["clients_feature_dict"] = clients_feature_dict
    config["clients_label_dict"] = clients_label_dict
    config["sample_to_client_assignment"] = sample_to_client_assignment
    
    return config


#
def distribute_concept_shift(config, quantile, incremental=False):
    # train/shared/test splitting
    shared_frac = config["shared_set_fraction"]
    test_frac = config["test_set_fraction"]

    is_regression = not config["num_classes"] > 1
    num_classes = config["num_classes"] if not is_regression else config["num_quantiles"]
    
    X_train, X_shared_and_test, y_train, y_shared_and_test = train_test_split(config["X_raw"], 
                                                                              config["y_raw"], 
                                                                              test_size = test_frac + shared_frac
                                                                             )
    X_test, X_shared, y_test, y_shared = train_test_split(X_shared_and_test, 
                                                          y_shared_and_test, 
                                                          test_size = shared_frac / (shared_frac + test_frac)
                                                         )
    del config["X_raw"]
    del config["y_raw"]

    if not incremental:
        quantile_val = np.quantile(np.array(y_train), quantile)
        is_below = np.array(y_train)<quantile_val
        num_below = int(quantile * config["num_clients"])

        X_below = np.array(X_train)[is_below]
        y_below = np.array(y_train)[is_below]
        X_above = np.array(X_train)[~is_below]
        y_above = np.array(y_train)[~is_below]
        X_train = np.append(X_below, X_above, axis=0)
        y_train = np.append(y_below, y_above, axis=0)
        sample_to_client_assignment = skews.apply_homogenous_data_distribution(X_below, y_below, num_below, config["num_classes"])
        clients_feature_dict, clients_label_dict = skews.apply_no_attribute_skew(X_below, y_below, sample_to_client_assignment)
        sample_to_client_assignment_tmp = skews.apply_homogenous_data_distribution(X_above, y_above, config["num_clients"] - num_below, config["num_classes"])
        sample_to_client_assignment_above = {key+len(sample_to_client_assignment): value for (key, value) in sample_to_client_assignment_tmp.items()}
        clients_feature_dict_above, clients_label_dict_above = skews.apply_no_attribute_skew(X_above, y_above, sample_to_client_assignment_above)
        
        sample_to_client_assignment_above_new = {}
        for key, value in sample_to_client_assignment_above.items():
            sample_to_client_assignment_above_new[key] = [x+len(y_below) for x in value]

        config["clients_feature_dict"] = clients_feature_dict | clients_feature_dict_above
        config["clients_label_dict"] = clients_label_dict | clients_label_dict_above
        config["sample_to_client_assignment"] = sample_to_client_assignment | sample_to_client_assignment_above_new

    else: 
        quantile_val = np.quantile(np.array(y_train), quantile)
        is_below = np.array(y_train)<quantile_val
        num_below = int(quantile * config["num_clients"])

        X_below = np.array(X_train)[is_below]
        y_below = np.array(y_train)[is_below]
        X_above = np.array(X_train)[~is_below]
        y_above = np.array(y_train)[~is_below]

        num_above = config["num_clients"] - num_below
        num_above_increment = int(num_above/2)
        num_above_no_increment = num_above - num_above_increment

        y_above_sorted, X_above_sorted = zip(*sorted(list(zip(y_above.tolist(), X_above.tolist()))))
        num_per_client = int(len(y_above)/num_above)

        X_train = np.append(X_below, X_above_sorted, axis=0)
        y_train = np.append(y_below, y_above_sorted, axis=0)

        y_above_increment = y_above_sorted[:num_per_client*num_above_increment]
        X_above_increment = X_above_sorted[:num_per_client*num_above_increment]
        y_above_no_increment = y_above_sorted[num_per_client*num_above_increment:]
        X_above_no_increment = X_above_sorted[num_per_client*num_above_increment:]

        # clients below
        sample_to_client_assignment = skews.apply_homogenous_data_distribution(X_below, y_below, num_below, config["num_classes"])
        clients_feature_dict, clients_label_dict = skews.apply_no_attribute_skew(X_below, y_below, sample_to_client_assignment)

        # clients incremental
        sample_ot_client_above_increment_tmp = {}
        for i in range(num_above_increment):
            sample_ot_client_above_increment_tmp[i] = list(range(i*num_per_client, (i+1)*num_per_client))
        sample_ot_client_above_increment = {key+len(sample_to_client_assignment): value for (key, value) in sample_ot_client_above_increment_tmp.items()}
        clients_feature_dict_above_increment, clients_label_dict_above_increment = skews.apply_no_attribute_skew(X_above_increment, y_above_increment, sample_ot_client_above_increment)
        sample_ot_client_above_increment_new = {}
        for key, value in sample_ot_client_above_increment.items():
            sample_ot_client_above_increment_new[key] = [x+len(y_below) for x in value]

        # clients no increment
        sample_ot_client_above_no_increment_tmp = skews.apply_homogenous_data_distribution(X_above_no_increment, y_above_no_increment, num_above_no_increment, config["num_classes"])
        sample_ot_client_above_no_increment = {key+len(sample_to_client_assignment)+len(sample_ot_client_above_increment_new): value for (key, value) in sample_ot_client_above_no_increment_tmp.items()}
        clients_feature_dict_above_no_increment, clients_label_dict_above_no_increment = skews.apply_no_attribute_skew(X_above_no_increment, y_above_no_increment, sample_ot_client_above_no_increment)
        sample_ot_client_above_no_increment_new = {}
        for key, value in sample_ot_client_above_no_increment.items():
            sample_ot_client_above_no_increment_new[key] = [x+len(y_below)+len(y_above_increment) for x in value]

        config["clients_feature_dict"] = clients_feature_dict | clients_feature_dict_above_increment | clients_feature_dict_above_no_increment
        config["clients_label_dict"] = clients_label_dict | clients_label_dict_above_increment | clients_label_dict_above_no_increment
        config["sample_to_client_assignment"] = sample_to_client_assignment | sample_ot_client_above_increment_new | sample_ot_client_above_no_increment_new

    config["X_train"] = X_train
    config["y_train"] = y_train
    config["X_test"] = X_test
    config["y_test"] = y_test
    config["X_shared"] = X_shared
    config["y_shared"] = y_shared
    config["num_below"] = num_below

    return config


# inserts a number of malicious users to the cohort
# takes the config
# returns the updated config
def insert_malicious_users(config):

    if "malicious_users" not in config:
        config["malicious_users"] = 0

    if "malicious_user_mode" not in config:
        config["malicious_user_mode"] = "label_zeros"

    updated_clients_feature_dict = config["clients_feature_dict"]
    updated_clients_label_dict = config["clients_label_dict"]
    updated_sample_to_client_assignment = config["sample_to_client_assignment"]
    updated_X_train = config["X_train"]
    updated_y_train = config["y_train"]

    for i in range(config["malicious_users"]):

        blueprint_client = random.randint(0, config["num_clients"]-1)
        malicious_user_idx = config["num_clients"] + i

        if config["malicious_user_mode"] == "label_zeros":

            updated_clients_feature_dict[malicious_user_idx] = updated_clients_feature_dict[blueprint_client]
            updated_clients_label_dict[malicious_user_idx] = list(np.zeros(len(updated_clients_label_dict[blueprint_client])).astype(int))

            updated_sample_to_client_assignment[malicious_user_idx] = list(range(len(updated_X_train), len(updated_X_train) + len(updated_clients_label_dict[blueprint_client])))

            updated_X_train.extend(updated_clients_feature_dict[blueprint_client])
            updated_y_train.extend(list(np.zeros(len(updated_clients_label_dict[blueprint_client])).astype(int)))

        elif config["malicious_user_mode"] == "label_introduction":

            m_client_features = updated_clients_feature_dict[blueprint_client]
            m_client_labels = np.array(updated_clients_label_dict[blueprint_client])

            max_idx = np.argmax(np.bincount(m_client_labels))
            indices = np.where(m_client_labels == max_idx)[0]
            np.put(m_client_labels, indices, config["num_classes"])
            m_client_labels = list(m_client_labels)

            updated_clients_feature_dict[malicious_user_idx] = m_client_features
            updated_clients_label_dict[malicious_user_idx] = m_client_labels

            updated_sample_to_client_assignment[malicious_user_idx] = list(range(len(updated_X_train), len(updated_X_train) + len(updated_clients_label_dict[blueprint_client])))

            updated_X_train.extend(m_client_features)
            updated_y_train.extend(m_client_labels)

        elif config["malicious_user_mode"] == "label_random":

            updated_clients_feature_dict[malicious_user_idx] = updated_clients_feature_dict[blueprint_client]
            updated_clients_label_dict[malicious_user_idx] = list(np.random.randint(config["num_classes"], size=len(updated_clients_label_dict[blueprint_client])).astype(int))

            updated_sample_to_client_assignment[malicious_user_idx] = list(range(len(updated_X_train), len(updated_X_train) + len(updated_clients_label_dict[blueprint_client])))

            updated_X_train.extend(updated_clients_feature_dict[blueprint_client])
            updated_y_train.extend(list(np.random.randint(config["num_classes"], size=len(updated_clients_label_dict[blueprint_client])).astype(int)))


        elif config["malicious_user_mode"] == "no_samples":

            updated_clients_feature_dict[malicious_user_idx] = updated_clients_feature_dict[blueprint_client][0:1]
            updated_clients_label_dict[malicious_user_idx] = updated_clients_label_dict[blueprint_client][0:1]

            updated_sample_to_client_assignment[malicious_user_idx] = list(range(len(updated_X_train), len(updated_X_train) + len(updated_clients_label_dict[malicious_user_idx])))

            updated_X_train.extend(updated_clients_feature_dict[blueprint_client][0:1])
            updated_y_train.extend(updated_clients_label_dict[blueprint_client][0:1])

        elif config["malicious_user_mode"] == "feature_zeros":

            s = np.array(updated_clients_feature_dict[blueprint_client]).shape

            updated_clients_feature_dict[malicious_user_idx] = list(np.zeros(s))
            updated_clients_label_dict[malicious_user_idx] = updated_clients_label_dict[blueprint_client]

            updated_sample_to_client_assignment[malicious_user_idx] = list(range(len(updated_X_train), len(updated_X_train) + len(updated_clients_label_dict[blueprint_client])))

            updated_X_train.extend(list(np.zeros(s)))
            updated_y_train.extend(updated_clients_label_dict[blueprint_client])

        else:
            
            print("malicious_user_mode \""+config["malicious_user_mode"]+"\" is not defined")

    if config["malicious_user_mode"] == "label_introduction":
        tmp = config["num_classes"] + 1
        config["num_classes"] = tmp

    config["sample_to_client_assignment"] = updated_sample_to_client_assignment
    config["X_train"] = updated_X_train
    config["y_train"] = updated_y_train
    config["clients_feature_dict"] = updated_clients_feature_dict
    config["clients_label_dict"] = updated_clients_label_dict
    config["num_clients"] = config["num_clients"] + config["malicious_users"]

    return config


# takes a population which is to be devided in quantiles and a list of values
# assignes quantiles to values in accordance with data distribution in population
# returns a list of quantile indices
def assign_quantiles(population, values, q = 4):
    quantiles = []
    for i in range(q+1):
        quantile = np.quantile(population, i/q)
        quantiles.append(quantile)
    quantiles[0] = min(population)-1
    quantiles[-1] = max(population)+1
    ret = []
    for value in values:
        for i, quantile in enumerate(quantiles[:-1]):
            quantile_plus = quantiles[i+1]
            if float(value) >= float(quantile) and float(value) < float(quantile_plus):
                ret.append(i)
                break
    return ret
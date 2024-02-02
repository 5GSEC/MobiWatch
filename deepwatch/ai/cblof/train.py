#!/usr/bin/python3.9

# Train an autoencoder-based DL model
import numpy as np
import torch
import sys
from pyod.models.cblof import CBLOF
from joblib import dump

# training dataset
train_dataset = "mobileinsight"
train_label = "benign"
train_ver = "v3"

# training parameter
params = {"n_clusters": 2, 
          "contamination": 0.1,
          "clustering_estimator": None,
          "alpha": 0.9,
          "beta": 5,
          "use_weights": False,
          "check_estimator": False,
          "random_state": False,
          "n_jobs": 1
          }

if __name__ == "__main__":
    # dataset_name = "%s_%s_%s.npy" % (train_dataset, train_label, train_ver)
    # train_feat = np.load('../../preprocessing/data/%s' % (dataset_name))
    # print(train_feat.shape)

    dataset_name = "%s_%s_%s.npy" % ("mobileinsight", "benign", train_ver)
    train_feat_mi = np.load('../../preprocessing/data/%s' % (dataset_name))

    dataset_name = "%s_%s_%s.npy" % ("phoenix", "abnormal", train_ver)
    train_feat_ph = np.load('../../preprocessing/data/%s' % (dataset_name))

    dataset_name = "%s_%s_%s.npy" % ("5g-spector", "abnormal", train_ver)
    train_feat_5s = np.load('../../preprocessing/data/%s' % (dataset_name))

    train_feat = np.append(train_feat_mi, train_feat_ph, axis=0)
    train_feat = np.append(train_feat, train_feat_5s, axis=0)

    print(train_feat.shape)

    params["contamination"] = (train_feat_ph.shape[0] + train_feat_5s.shape[0]) / train_feat.shape[0]
    print(params["contamination"])

    model = CBLOF(n_clusters=params["n_clusters"], 
                  contamination=params["contamination"],
                  clustering_estimator=params["clustering_estimator"],
                  alpha=params["alpha"],
                  beta=params["beta"],
                  use_weights=params["use_weights"],
                  check_estimator=params["check_estimator"],
                  random_state=params["random_state"],
                  n_jobs=params["n_jobs"]
                  )
    
    model.fit(train_feat)
    dump(model, "./save/cblof_%s.joblib" % train_ver)



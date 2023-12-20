# make a function that collects all the words from the two
from time import time
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch import tensor as tensor
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

lds_embeds = pd.read_csv(Path('final_project/lds_red_embeds.csv'), index_col=0)
christian_embeds = pd.read_csv(Path('final_project/christian_red_embeds.csv'), index_col=0)



# repentance, church, gospel, bishop, covenant, repent, scripture, sabbath, apostle, 
# prophet, bishop, bless, redeemer, mormon, salvation, resurrection, sacrament, jesus

keywords = ["repentance", "jesus", "church", "mormon", "gospel", "sacrament", "bishop", "covenant","repent", "scripture", "sabbath", "apostle",
            "prophet", "bishop", "bless", "redeemer", "salvation", "resurrection", "savior", "preach", "satan", "priesthood",
            "commandment", "saints", "christ", "joseph", "temple", "baptism", "exalt", "godhead", "atonement", "apostasy", "stake",
            "ordinance", "sin", "preach", "holiness", "apostles", "mormons", "gentiles", "disciple", "baptism", "baptized",
            "revelation", "catholic", "kingdom", "celestial", "patriarch", "missionary"]

for word in keywords:
    try:
        keyword_lds_embeds = lds_embeds.loc[lds_embeds['token'] == word.lower()]
        keyword_christian_embeds = christian_embeds.loc[christian_embeds['token'] == word.lower()]
        print('here')

        keyword_lds_embeds.insert(len(keyword_lds_embeds.columns), 'label', 'lds', allow_duplicates=True)
        keyword_christian_embeds.insert(len(keyword_christian_embeds.columns), 'label', 'christian', allow_duplicates=True)
        keyword_embeds_df = pd.concat([keyword_lds_embeds, keyword_christian_embeds], axis=0)
        # need to convert the tensor to numpy
        print(keyword_embeds_df.head)

        X = np.array([eval(embedding).numpy() for embedding in keyword_embeds_df['embedding']])
        print(X)
        print(X.shape)
        y = keyword_embeds_df['label'].to_numpy()
        print(y)
        print(y.shape)

        pca = PCA(n_components=2, svd_solver='auto')
        pca = pca.fit(X)
        x_new = pca.transform(X)

        # all_new_np = np.concatenate((x_new, y), axis=1)
        colors = {'lds': 'tab:blue', 'christian': 'tab:red'}

        plt.scatter(x_new[:,0], x_new[:,1], c = keyword_embeds_df['label'].map(colors))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig(Path(f'final_project/pca_plots/pca_{word}.png'))
        # plt.show()
        plt.clf()
    except Exception as e:
        with open(Path("final_project/pca_plots/errors.txt"), "a") as f:
            print(type(e), e, f"{word}", file=f)
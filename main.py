from utils import load_embeddings
from argparse import Namespace
import time
import torch
from sklearn.decomposition import PCA
from nltk.corpus import brown
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    params = Namespace()
    params.src_lang = 'en'
    params.src_emb = './wiki.en.pth'
    params.max_vocab_A = 0
    params.dataset = 'muse'
    params.emb_dim = 300
    params.cuda = False

    dico, embeddings = load_embeddings(params,True,True)
    pca = PCA(2)
    print(dico.index('.'))

    t1 = time.time()
    traces = []
    indices = set()
    for sent in brown.sents()[:5]:
        print(sent)
        idx_list = []
        for word in sent:
            if word in dico and word!='``':
                idx = dico.index(word.lower())
                idx_list.append(idx)
        if len(idx_list) > 0:
            indices = indices.union(idx_list)
            traces.append(embeddings[idx_list].numpy())
    print(time.time()-t1)

    for idx in indices:
        print(dico[idx])
    # all_emb2 = np.concatenate(traces,0)
    # pca.fit(all_emb2)
    
    all_emb = embeddings[list(indices)].numpy()
    all_emb_2d = pca.fit_transform(all_emb)

    t1 = time.time()
    for i,idx in enumerate(list(indices)):
        plt.text(all_emb_2d[i,0],all_emb_2d[i,1],s=dico[idx])
        plt.scatter(all_emb_2d[i,0],all_emb_2d[i,1],marker='.')
    for sent in traces:
        sent = pca.transform(sent)
        plt.plot(sent[:,0],sent[:,1])
    print(time.time()-t1)
    plt.show()
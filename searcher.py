import faiss
import time
import numpy as np
from sklearn.preprocessing import normalize


class Searcher():
    def __init__(self, ngpu=1, feat_len=256):
        self._init_faiss(ngpu, feat_len)

    def _init_faiss(self, ngpu, feat_len):
        self.flat_config = []
        for i in range(ngpu):
            self.cfg = faiss.GpuIndexFlatConfig()
            self.cfg.useFloat16 = False
            self.cfg.device = i
            self.flat_config.append(self.cfg)
        self.res = [faiss.StandardGpuResources() for i in range(ngpu)]
        self.indexes = [faiss.GpuIndexFlatL2(self.res[i], feat_len, self.flat_config[i]) for i in range(ngpu)]
        self.index = faiss.IndexProxy()
        for sub_index in self.indexes:
            self.index.addIndex(sub_index)

    def search_by_topk(self, Query, Gallery, topk=20, if_normalize=True):
        if if_normalize:
            Gallery = normalize(Gallery, norm='l2').astype('float32')
            Query   = normalize(Query, norm='l2').astype('float32')
        self.index.reset()
        self.index.add(Gallery)
        topk_scores, topk_idx = self.index.search(Query, topk)
        return topk_scores, topk_idx


class Searcher2():
    def __init__(self, ngpu=1, feat_len=256):
        self._init_faiss(ngpu, feat_len)

    def _init_faiss(self, ngpu, feat_len):
        self.flat_config = []
        for i in range(ngpu):
            self.cfg = faiss.GpuIndexFlatConfig()
            self.cfg.useFloat16 = False
            self.cfg.device = i
            self.flat_config.append(self.cfg)
        self.res = [faiss.StandardGpuResources() for i in range(ngpu)]
        self.indexes = [faiss.GpuIndexFlatIP(self.res[i], feat_len, self.flat_config[i]) for i in range(ngpu)]
        self.index = faiss.IndexProxy()
        for sub_index in self.indexes:
            self.index.addIndex(sub_index)

    def search_by_topk(self, Query, Gallery, topk=20, if_normalize=True):
        if if_normalize:
            #Gallery = normalize(Gallery, norm='l2').astype('float32')
            Query   = normalize(Query, norm='l2').astype('float32')
        #self.index.reset()
        #self.index.add(Gallery)
        topk_scores, topk_idx = self.index.search(Query, topk)
        return topk_scores, topk_idx

    def add_feature(self, feature):
        feature = normalize(feature, norm='l2').astype('float32')
        self.index.add(feature)


class PQSearcher():
    def __init__(self, ngpu=1, feat_len=256, index='32', tempIVF="IVF20000", tempPQ="PQ32", tempNP=100):
        self._init_faiss(ngpu, feat_len, tempIVF, tempPQ, tempNP)

    def _init_faiss(self, ngpu, feat_len, tempIVF, tempPQ, tempNP):
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.usePrecomputed = False

        # Setting up GPU resources
        self.res = [faiss.StandardGpuResources() for i in range(ngpu)]
        self.indexes = []
        for i in range(ngpu):
            index = faiss.index_factory(feat_len,tempIVF + "," + tempPQ)
            index.nprobe = tempNP
            index = faiss.index_cpu_to_gpu(self.res[i],i,index,co)
            self.indexes.append(index)
        self.index = faiss.IndexProxy()
        for sub_index in self.indexes:
            self.index.addIndex(sub_index)

    def search_by_topk(self, features, Gallery, topk=20, if_normalize=True):
        if if_normalize:
            features = normalize(features, norm='l2').astype('float32')
        self.index.train(Gallery)
        self.index.add(Gallery)
        topk_scores, topk_idx = self.index.search(features, topk+1)
        return topk_scores, topk_idx

    def getIndex(self, features, if_normalize=True, tempDir="./PQindex/feature_index", tempIVF="IVF1024", tempPQ="PQ32", tempNP=100):
        if if_normalize:
            features = normalize(features, norm='l2').astype('float32')
        index = faiss.index_factory(features.shape[1], tempIVF + "," + tempPQ)
        index.train(features)
        faiss.write_index(index, tempDir+tempIVF+tempPQ+"nprobe"+str(tempNP)+".bin")

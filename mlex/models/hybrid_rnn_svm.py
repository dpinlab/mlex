from mlex.models.base_components.hybrid import HybridRecurrentSVMBase, HybridSVM


class hybrid_rnn_svm(HybridSVM):
    BASE_MODEL_CLS = HybridRecurrentSVMBase
    NAME = 'hybrid_rnn_svm'

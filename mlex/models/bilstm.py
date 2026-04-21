from mlex.models.base_components.recurrent import BILSTMBaseModel, RecurrentModel


class BILSTM(RecurrentModel):
    BASE_MODEL_CLS = BILSTMBaseModel
    NAME = 'BILSTM'

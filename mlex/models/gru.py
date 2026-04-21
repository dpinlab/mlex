from mlex.models.base_components.recurrent import GRUBaseModel, RecurrentModel


class GRU(RecurrentModel):
    BASE_MODEL_CLS = GRUBaseModel
    NAME = 'GRU'

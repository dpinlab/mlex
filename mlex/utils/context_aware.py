import pandas as pd

class ContextAware:
    def __init__(self, target_column: str, timestamp_column: str, context_column:str):
        """
        Classe responsável por realizar operações sensíveis ao contexto temporal e de grupo.

        Parâmetros:
        - target_column (str): nome da coluna alvo em y.
        - timestamp_column (str): nome da coluna de tempo usada para ordenação.
        """
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.context_column = context_column

    def transform(self, X: pd.DataFrame, y: pd.DataFrame):



        """
        Ordena os dataframes X e y com base nas colunas 'GROUP' e timestamp_column.

        Retorna:
        - (X_ordenado, y_ordenado)
        """
        
        if self.context_column:
            X['GROUP'] = X[self.context_column].fillna('Unknown')
        else:
            X['GROUP'] = 'Unknown'

        df = X.copy()
        df['_y'] = y[self.target_column].values

        # Ordenação por contexto
        df = df.sort_values(by=['GROUP', self.timestamp_column]).reset_index(drop=True)

        # Separar novamente X e y
        new_y = df[['_y']].rename(columns={'_y': self.target_column})
        df.drop(columns=['_y'], inplace=True)

        return df, new_y

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

## SEED

_seed = 2118

## DEFINING COLUMNS

colunas_para_normalizar = ['age', 'water_ingestion_per_day', 'exercise_frequency', 'eat_vegetables', 'how_many_meals_per_day', 'time_under_eletronics']
colunas_para_transpor = ['eat_between_meals', 'drink_frequency', 'gender']
colunas_para_excluir = ['imc', 'height', 'weight', 'sk_eat_between_meals', 'sk_drink_frequency', 'sk_gender']

## CLASS DROPPING COLUMNS

class DroppingColumns(BaseEstimator, TransformerMixin):

    def __init__(self, ToBe_Dropped=colunas_para_excluir):
        self.ToBe_Dropped = ToBe_Dropped

    def fit(self, df):
        return self
    
    def transform(self, df):
        df_tobe_dropped = df.copy()
        df_tobe_dropped.drop(self.ToBe_Dropped, axis=1, inplace=True)
        return df_tobe_dropped

## CLASS STANDARD SCALER
    
class StdScaler(BaseEstimator, TransformerMixin):

    def __init__(self, ToBe_Scaled=colunas_para_normalizar):
        self.ToBe_Scaled = ToBe_Scaled

    def fit(self, df):
        return self
    
    def transform(self, df):
        std_Scaler = StandardScaler()
        df_tobe_scaled = df.copy()
        df_tobe_scaled[self.ToBe_Scaled] = std_Scaler.fit_transform(df_tobe_scaled[self.ToBe_Scaled])
        return df_tobe_scaled

## CLASS ONE HOT ENCODER
   
class OHEnconder(BaseEstimator, TransformerMixin):

    def __init__(self, ToBe_Transposed=colunas_para_transpor):
        self.ToBe_Transposed = ToBe_Transposed

    def fit(self, df):
        return self
    
    def transform(self, df):
        
        def onehot_encoder(df, ToBe_Transposed):
            oh_enc = OneHotEncoder()
            oh_enc.fit(df[ToBe_Transposed])
            names = oh_enc.get_feature_names_out(ToBe_Transposed)
            df = pd.DataFrame(
                oh_enc.transform(df[self.ToBe_Transposed]).astype(int).toarray()
                ,columns=names
                ,index=df.index
            )
            return df
        
        def concatenate(df, oh_enc_df, ToBe_Transposed):
            other_features = [x for x in df.columns if x not in ToBe_Transposed]
            df_concat = pd.concat(
                [oh_enc_df, df[other_features]]
                ,axis=1
            )
            return df_concat
        
        df_OneHotEncoded = onehot_encoder(df, self.ToBe_Transposed)

        df_full = concatenate(df, df_OneHotEncoded, self.ToBe_Transposed)

        return df_full
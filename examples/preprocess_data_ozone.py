import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pandas as pd
import numpy as np

df = pd.read_excel('/data/ozone/ozone.xlsx')
df_limpo = df.replace('?', np.nan).dropna()

print(f"Linhas antes: {len(df)}")
print(f"Linhas depois: {len(df_limpo)}")
print(df_limpo.head())
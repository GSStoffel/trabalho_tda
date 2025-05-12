import json
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib import ticker
from sklearn.linear_model import LinearRegression

ano_atual = datetime.now().year
qtd_anos = 20
ultimos_anos = [i for i in range(ano_atual - qtd_anos, ano_atual + 1)]
ultimos_anos_str = [str(i) for i in range(ano_atual - qtd_anos, ano_atual + 1)]

anos_pipe = '|'.join(ultimos_anos_str)
url_pib = f'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/{anos_pipe}/variaveis/37?localidades=N3[all]'
response_pib = requests.get(url_pib)

data_pib = response_pib.json()[0]['resultados'][0]['series']
df_pib = pd.read_json(StringIO(json.dumps(data_pib)), orient='records')

df_pib['estado'] = df_pib['localidade'].apply(lambda row: row['nome'])

for ano in ultimos_anos:
  df_pib[ano] = df_pib['serie'].apply(lambda row: pd.to_numeric(row.get(str(ano), float('nan'))) * 100)

df_pib = df_pib.drop(columns=['localidade', 'serie'])

siglas = {
    'Acre': 'AC',
    'Alagoas': 'AL',
    'Amapá': 'AP',
    'Amazonas': 'AM',
    'Bahia': 'BA',
    'Ceará': 'CE',
    'Distrito Federal': 'DF',
    'Espírito Santo': 'ES',
    'Goiás': 'GO',
    'Maranhão': 'MA',
    'Mato Grosso': 'MT',
    'Mato Grosso do Sul': 'MS',
    'Minas Gerais': 'MG',
    'Pará': 'PA',
    'Paraíba': 'PB',
    'Paraná': 'PR',
    'Pernambuco': 'PE',
    'Piauí': 'PI',
    'Rio de Janeiro': 'RJ',
    'Rio Grande do Norte': 'RN',
    'Rio Grande do Sul': 'RS',
    'Rondônia': 'RO',
    'Roraima': 'RR',
    'Santa Catarina': 'SC',
    'São Paulo': 'SP',
    'Sergipe': 'SE',
    'Tocantins': 'TO'
}
df_pib['SIGLA'] = df_pib['estado'].map(siglas)

df_pib.drop(columns=['estado'], inplace=True)
df_pib = df_pib.dropna(axis=1)

# Transpondo Anos para coluna ano
df_pib = df_pib.melt(id_vars=['SIGLA'], var_name='ANO', value_name='VALOR')

# Ano para inteiros
df_pib['ANO'] = df_pib['ANO'].astype(int)

#####################

# Ordernação por valor e estado
df_pib = df_pib.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])

# Configurando estilo
sns.set_style(style="whitegrid")

# Configurando dimenções
plt.figure(figsize=(24, 10))

# Criando as barras
sns.barplot(x='SIGLA', y='VALOR', hue='ANO', data=df_pib)

# Rótulos e título
plt.xlabel('Estado')
plt.ylabel('Valor (em bilhões)')
plt.title('Comparação PIB entre estados')

# Plotando
plt.tight_layout()
plt.savefig('output/comparacao_pib_estados.png')
plt.close()
#####################

for ano in ultimos_anos:
    df_pib_ano = df_pib[df_pib['ANO'] == ano]

    # Ordernação por valor e estado
    df_pib_ano = df_pib_ano.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])

    # Configurando estilo
    sns.set_style(style="whitegrid")

    # Configurando dimenções
    plt.figure(figsize=(16, 10))

    # Criando as barras
    sns.barplot(x='SIGLA', y='VALOR', data=df_pib_ano)

    # Rótulos e título
    plt.xlabel('Estado')
    plt.ylabel('Valor (em bilhões)')
    plt.title(f'Comparação PIB entre estados ({ano})')

    # Plotando
    plt.tight_layout()
    plt.savefig(f'output/comparacao_pib_estados_{ano}.png')
    plt.close()
#####################

file_id = '1xc40YIHHr_d9kQWjZ9i5eLE_OVzI701c'
url = f'https://drive.google.com/uc?export=download&id={file_id}'

response = requests.get(url)
with open('populacao.xlsx', 'wb') as f:
    f.write(response.content)

df_pop = pd.read_excel('populacao.xlsx', header=1, skiprows=4, engine='openpyxl')

df_pop = df_pop[(df_pop['SEXO'] == 'Ambos') & (~df_pop['SIGLA'].isin(['CO', 'ND', 'NO', 'SD', 'SU', 'BR']))]
df_pop = df_pop.drop(columns=['IDADE', 'CÓD.', 'SEXO', 'LOCAL'])
# Agregando por sigla
df_pop = df_pop.groupby('SIGLA', as_index=False).sum()
# Transpondo Anos para coluna ano
df_pop = df_pop.melt(id_vars=['SIGLA'], var_name='ANO', value_name='VALOR')
# Ano para inteiro
df_pop['ANO'] = df_pop['ANO'].astype(int)
# Anos até 2025
df_pop = df_pop[df_pop['ANO'] <= ano_atual]

#####################

# Ordernação por valor e estado
df_pop = df_pop.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])

# Configurando estilo
sns.set_style(style="whitegrid")

# Configurando dimenções
plt.figure(figsize=(24, 10))

# Criando as barras
sns.barplot(x='SIGLA', y='VALOR', hue='ANO', data=df_pop)

# Rótulos e título
plt.xlabel('Estado')
plt.ylabel('Número de Pessoas')
plt.title('Comparação da população entre estados')

# Plotando
plt.tight_layout()
plt.savefig(f'output/comparacao_pop_estados.png')
plt.close()
######################

for ano in ultimos_anos:
    df_pop_ano = df_pop[df_pop['ANO'] == ano]

    # Ordernação por valor e estado
    df_pop_ano = df_pop_ano.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])

    # Configurando estilo
    sns.set_style(style="whitegrid")

    # Configurando dimenções
    plt.figure(figsize=(16, 10))

    # Criando as barras
    sns.barplot(x='SIGLA', y='VALOR', data=df_pop_ano)

    # Rótulos e título
    plt.xlabel('Estado')
    plt.ylabel('Número de Pessoas')
    plt.title('Comparação da população entre estados (2021)')

    # Plotando
    plt.tight_layout()
    plt.savefig(f'output/comparacao_pop_estados_{ano}.png')
    plt.close()
######################

# dividir os valores da tabela pib pelos valores da tabela população, resultando na criação de uma tabela de pib per capta
tabela_pib_per_capta = pd.merge(df_pib, df_pop, on=['SIGLA', 'ANO'], how='inner', suffixes=('_PIB', '_POP'))
tabela_pib_per_capta['PIB_PER_CAPTA'] = tabela_pib_per_capta['VALOR_PIB'] / tabela_pib_per_capta['VALOR_POP']

######################

# Ordernação por valor e estado
tabela_pib_per_capta = tabela_pib_per_capta.sort_values(by=['PIB_PER_CAPTA', 'SIGLA'], ascending=[False, True])

# Configurando estilo
sns.set_style(style="whitegrid")

# Configurando dimenções
plt.figure(figsize=(28, 10))

# Criando as barras
sns.barplot(x='SIGLA', y='PIB_PER_CAPTA', hue='ANO', data=tabela_pib_per_capta)

# Rótulos e título
plt.xlabel('Estados')
plt.ylabel('PIB per capta')
plt.title('PIB per capta')

# Plotando
plt.tight_layout()

########################

tabela_pib_per_capta_rs = tabela_pib_per_capta[tabela_pib_per_capta['SIGLA'] == 'RS']
X = tabela_pib_per_capta_rs['ANO'].values.reshape(-1, 1)
y = tabela_pib_per_capta_rs['PIB_PER_CAPTA'].values

lr = LinearRegression()
lr.fit(X, y)
y_predicts = lr.predict(X)

anos_futuros = np.array([2022, 2023, 2024, 2025]).reshape(-1, 1)
previsoes = lr.predict(anos_futuros)

######################

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_predicts, color='orange', label='Regressão Linear')
plt.plot(anos_futuros, previsoes, color='green', linestyle='--', marker='o', label='Previsão até 2025')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel('Ano')
plt.ylabel('PIB per capita')
plt.title('PIB per capita - RS (2005–2025)')
plt.legend()
plt.grid(True)
plt.savefig(f'output/previsao_pib_per_capta.png')
plt.close()

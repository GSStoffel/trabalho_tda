import json
import os
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import requests
import seaborn as sns
from matplotlib import ticker
from sklearn.linear_model import LinearRegression

ANO_ATUAL = datetime.now().year
QTD_ANOS = 20
ULTIMOS_ANOS = [i for i in range(ANO_ATUAL - QTD_ANOS, ANO_ATUAL + 1)]
SIGLAS = {
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

db_user = os.getenv('DB_USER', 'user')
db_password = os.getenv('DB_PASSWORD', 'password')
db_name = os.getenv('DB_NAME', 'meu_banco')
db_port = os.getenv('DB_PORT', '5432')
db_host = os.getenv('DB_HOST', '5432')


def main():
    data_pib = obtem_dados_pib()
    df_pib = transforma_dados_pib(data_pib)
    carrega_dados_pib(df_pib)
    plotar_graficos_pib(df_pib)

    obtem_dados_populacao()
    df_pop = transformar_dados_populacao()
    carrega_dados_populacao(df_pop)
    plotar_graficos_populacao(df_pop)

    tabela_pib_per_capta = calcular_pib_per_capta(df_pib, df_pop)
    carrega_dados_pib_per_capta(tabela_pib_per_capta)
    plotar_graficos_pib_per_capta(tabela_pib_per_capta)


def plotar_graficos_pib_per_capta(tabela_pib_per_capta):
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

    for sigla in tabela_pib_per_capta['SIGLA']:
        tabela_pib_per_capta_estado = tabela_pib_per_capta[tabela_pib_per_capta['SIGLA'] == sigla]
        X = tabela_pib_per_capta_estado['ANO'].values.reshape(-1, 1)
        y = tabela_pib_per_capta_estado['PIB_PER_CAPTA'].values

        lr = LinearRegression()
        lr.fit(X, y)
        y_predicts = lr.predict(X)

        anos_futuros = np.array([2022, 2023, 2024, 2025]).reshape(-1, 1)
        previsoes = lr.predict(anos_futuros)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Dados reais')
        plt.plot(X, y_predicts, color='orange', label='Regressão Linear')
        plt.plot(anos_futuros, previsoes, color='green', linestyle='--', marker='o', label='Previsão até 2025')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel('Ano')
        plt.ylabel('PIB per capita')
        plt.title('PIB per capita - RS (2022–2025)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/previsao_pib_per_capta_{sigla}.png')
        plt.close()


def calcular_pib_per_capta(df_pib, df_pop):
    # dividir os valores da tabela pib pelos valores da tabela população, resultando na criação de uma tabela de pib per capta
    tabela_pib_per_capta = pd.merge(df_pib, df_pop, on=['SIGLA', 'ANO'], how='inner', suffixes=('_PIB', '_POP'))
    tabela_pib_per_capta['PIB_PER_CAPTA'] = tabela_pib_per_capta['VALOR_PIB'] / tabela_pib_per_capta['VALOR_POP']
    return tabela_pib_per_capta


def plotar_graficos_populacao(df_pop):
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
    for ano in ULTIMOS_ANOS:
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


def transformar_dados_populacao():
    df_pop = pd.read_excel('data/populacao.xlsx', header=1, skiprows=4, engine='openpyxl')
    df_pop = df_pop[(df_pop['SEXO'] == 'Ambos') & (~df_pop['SIGLA'].isin(['CO', 'ND', 'NO', 'SD', 'SU', 'BR']))]
    df_pop = df_pop.drop(columns=['IDADE', 'CÓD.', 'SEXO', 'LOCAL'])
    # Agregando por sigla
    df_pop = df_pop.groupby('SIGLA', as_index=False).sum()
    # Transpondo Anos para coluna ano
    df_pop = df_pop.melt(id_vars=['SIGLA'], var_name='ANO', value_name='VALOR')
    # Ano para inteiro
    df_pop['ANO'] = df_pop['ANO'].astype(int)
    # Anos até 2025
    df_pop = df_pop[df_pop['ANO'] <= ANO_ATUAL]
    return df_pop


def obtem_dados_populacao():
    file_id = '1xc40YIHHr_d9kQWjZ9i5eLE_OVzI701c'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(url)
    with open('data/populacao.xlsx', 'wb') as f:
        f.write(response.content)


def plotar_graficos_pib(df_pib):
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
    for ano in ULTIMOS_ANOS:
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


def transforma_dados_pib(data_pib):
    df_pib = pd.read_json(StringIO(json.dumps(data_pib)), orient='records')
    df_pib['estado'] = df_pib['localidade'].apply(lambda row: row['nome'])
    for ano in ULTIMOS_ANOS:
        df_pib[ano] = df_pib['serie'].apply(lambda row: pd.to_numeric(row.get(str(ano), float('nan'))) * 100)
    df_pib = df_pib.drop(columns=['localidade', 'serie'])
    df_pib['SIGLA'] = df_pib['estado'].map(SIGLAS)
    df_pib.drop(columns=['estado'], inplace=True)
    df_pib = df_pib.dropna(axis=1)
    # Transpondo Anos para coluna ano
    df_pib = df_pib.melt(id_vars=['SIGLA'], var_name='ANO', value_name='VALOR')
    # Ano para inteiros
    df_pib['ANO'] = df_pib['ANO'].astype(int)
    return df_pib


def obtem_dados_pib():
    anos_pipe = '|'.join(map(str, ULTIMOS_ANOS))
    url_pib = f'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/{anos_pipe}/variaveis/37?localidades=N3[all]'
    response_pib = requests.get(url_pib)
    data_pib = response_pib.json()[0]['resultados'][0]['series']
    return data_pib


def carrega_dados_pib(df_pib):
    try:
        # Conexão com o banco de dados PostgreSQL
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        # Criação da tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tabela_pib (
                sigla VARCHAR(2),
                ano INT,
                valor NUMERIC,
                PRIMARY KEY (sigla, ano)
            );
        """)

        # Inserção dos dados no banco
        for _, row in df_pib.iterrows():
            cursor.execute("""
                INSERT INTO tabela_pib (sigla, ano, valor)
                VALUES (%s, %s, %s)
                ON CONFLICT (sigla, ano) DO UPDATE
                SET valor = EXCLUDED.valor;
            """, (row['SIGLA'], row['ANO'], row['VALOR']))

        # Commit e fechamento da conexão
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao carregar dados no PostgreSQL: {e}")


def carrega_dados_populacao(df_pop):
    try:
        # Conexão com o banco de dados PostgreSQL
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        # Criação da tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tabela_pop (
                sigla VARCHAR(2),
                ano INT,
                valor NUMERIC,
                PRIMARY KEY (sigla, ano)
            );
        """)

        # Inserção dos dados no banco
        for _, row in df_pop.iterrows():
            cursor.execute("""
                INSERT INTO tabela_pop (sigla, ano, valor)
                VALUES (%s, %s, %s)
                ON CONFLICT (sigla, ano) DO UPDATE
                SET valor = EXCLUDED.valor;
            """, (row['SIGLA'], row['ANO'], row['VALOR']))

        # Commit e fechamento da conexão
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao carregar dados no PostgreSQL: {e}")

def carrega_dados_pib_per_capta(tabela_pib_per_capta):
    try:
        # Conexão com o banco de dados PostgreSQL
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        # Criação da tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tabela_pib_per_capta (
                sigla VARCHAR(2),
                ano INT,
                valor NUMERIC,
                PRIMARY KEY (sigla, ano)
            );
        """)

        # Inserção dos dados no banco
        for _, row in tabela_pib_per_capta.iterrows():
            cursor.execute("""
                INSERT INTO tabela_pib_per_capta (sigla, ano, valor)
                VALUES (%s, %s, %s)
                ON CONFLICT (sigla, ano) DO UPDATE
                SET valor = EXCLUDED.valor;
            """, (row['SIGLA'], row['ANO'], row['PIB_PER_CAPTA']))

        # Commit e fechamento da conexão
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao carregar dados no PostgreSQL: {e}")


if __name__ == '__main__':
    main()

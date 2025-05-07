import requests
import pandas as pd
import json
from io import StringIO

ano_atual = 2020
ultimos_5_anos = [str(i) for i in range(ano_atual - 4, ano_atual + 1)]
anos = '|'.join([str(i) for i in range(ano_atual - 5, ano_atual + 1)])
url_pib = 'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/{anos}/variaveis/37?localidades=N3[all]'
url_pib = url_pib.format(anos=anos)


def extract():
    global df_pib
    df_pib['estado'] = df_pib['localidade'].apply(lambda row: row['nome'])
    df_pib['2019'] = df_pib['serie'].apply(lambda row: pd.to_numeric(row['2019']))
    for ano in ultimos_5_anos:
        df_pib[ano] = df_pib['serie'].apply(lambda row: pd.to_numeric(row.get(ano, float('nan'))))
    
    # Remoção das colunas originais pós extração    
    df_pib = df_pib.drop(columns=['localidade', 'serie'])




def load():
    global data_pib
    global df_pib
    response_pib = requests.get(url_pib)
    data_pib = response_pib.json()[0]['resultados'][0]['series']
    df_pib = pd.read_json(StringIO(json.dumps(data_pib)), orient='records')    


def transform():
    #Salvar o df_pib no banco de dados
    global df_pib
    
    from sqlalchemy import create_engine
    
    # Configuração de conexão
    DATABASE_URL = "postgresql://user:password@postgres:5432/meu_banco"
    
    try:
        # Cria a engine de conexão
        engine = create_engine(DATABASE_URL)
        
        # Salva o DataFrame na tabela 'pib'
        df_pib.to_sql(
            name='pib',
            con=engine,
            if_exists='replace',  # Substitui a tabela se existir
            index=False
        )
        print("Dados salvos com sucesso no PostgreSQL!")
        
    except Exception as e:
        print(f"Erro ao salvar dados: {e}")


def main():
    # Carrega os dados
    load()

    # Processa os dados
    extract()

    # Salva no banco
    transform()
    

if __name__ == "__main__":
    main()

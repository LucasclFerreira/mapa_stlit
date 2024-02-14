import json
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px

# -------------------- CONFIGURAÇÕES ----------------------
titulo_pagina = 'Mapa de Desastres Climáticos'
layout = 'wide'
st.set_page_config(page_title=titulo_pagina, layout=layout)
st.title(titulo_pagina)
# ---------------------------------------------------------



# FUNÇÕES
@st.cache_data
def carrega_geojson(caminho):
    with open(caminho, 'r') as f:
        geoj = json.load(f)
    return geoj

@st.cache_data
def filtra_geojson(geojson, iso, prop='codarea'):
    gdf = gpd.GeoDataFrame.from_features(geojson)
    return json.loads(gdf[gdf[prop] == iso].to_json())

@st.cache_data
def carrega_dados(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    return df

@st.cache_data
def carrega_parquet(caminho_arquivo):
    df = pd.read_parquet(caminho_arquivo)
    return df

@st.cache_data
def carrega_malha(tipo='estados', uf='MG', intrarregiao='municipio', qualidade='minima'):
    url = f'https://servicodados.ibge.gov.br/api/v3/malhas/{tipo}/{uf}?formato=application/vnd.geo+json&intrarregiao={intrarregiao}&qualidade={qualidade}'
    return requests.get(url).json()

def filtra_estado(df, uf):
    return df[(df.uf.eq(uf))]

def filtra_grupo_desastre(df, grupo_desastre):
    return df[df.grupo_de_desastre == grupo_desastre]

def filtra_ano(df, inicio, fim):
    return df[(df.data.ge(f'{inicio}-01-01')) & (df.data.le(f'{fim}-12-30'))]

def calcula_ocorrencias(df, cols_selecionadas, cols_agrupadas):
    return df.groupby(cols_agrupadas, as_index=False)[cols_selecionadas].count().rename(columns={'protocolo': 'ocorrencias'})

def classifica_risco(df, col_ocorrencias):
    quartis = df[col_ocorrencias].quantile([0.2, 0.4, 0.6, 0.8]).values
    risco = []
    for valor in df[col_ocorrencias]:
        if valor > quartis[3]:
            risco.append('Muito Alto')
        elif valor > quartis[2]:
            risco.append('Alto')
        elif valor > quartis[1]:
            risco.append('Moderado')
        elif valor > quartis[0]:
            risco.append('Baixo')
        else:
            risco.append('Muito Baixo')
    df['risco'] = risco
    return df

def classifica_segurado(base, dataframe):
    df = dataframe.copy()
    nenhum = set(base.query("descricao_tipologia == 'Nenhum'").ibge)
    algum = set(base.query("descricao_tipologia != 'Nenhum'").ibge)
    resultado = list(nenhum - algum)
    df['seg'] = np.where(df.ibge == '-', 'nao segurado', 'segurado com sinistro')
    df.seg = df.seg.mask(df.code_muni.isin(resultado), 'segurado sem sinistro')
    return df

def cria_mapa(df, malha, locais='ibge', cor='ocorrencias', tons=None, tons_midpoint=None, nome_hover=None, dados_hover=None, lista_cores=None, lat=-14, lon=-53, zoom=3, titulo_legenda='Risco', featureid='properties.codarea'):
    fig = px.choropleth_mapbox(
        df, geojson=malha, color=cor,
        color_continuous_scale=tons,
        color_continuous_midpoint=tons_midpoint,
        color_discrete_map=lista_cores,
        category_orders={cor: list(lista_cores.keys())},
        labels={'risco': 'Risco', 'ocorrencias': 'Ocorrências', 'code_muni': 'Código Municipal',
                 'code_state': 'Código', 'desastre_mais_comum': 'Desastre mais comum'},
        locations=locais, featureidkey=featureid,
        center={'lat': lat, 'lon': lon}, zoom=zoom, 
        mapbox_style='carto-positron', height=500,
        hover_name=nome_hover, hover_data=dados_hover,
        opacity=0.95
    )

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_bounds={"west": -150, "east": -20, "south": -60, "north": 60},
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgb(250, 250, 250)',
            font=dict(size=14),
            title=dict(
                font=dict(size=16),
                text=titulo_legenda
            ),
            traceorder="normal"
        )
    )
    
    return fig



# VARIAVEIS
dados_atlas = carrega_parquet('desastres_latam2.parquet')
dados_merge = carrega_parquet('area2.parquet')
coord_uf = carrega_parquet('coord_uf.parquet')
coord_latam = carrega_parquet('coord_latam3.parquet')
coord_muni = carrega_parquet('coord_muni.parquet')
pop_pib = carrega_parquet('pop_pib_muni.parquet')
pop_pib_uf = carrega_parquet('pop_pib_latam.parquet')
malha_america = carrega_geojson('malha_latam.json')
malha_brasil = carrega_geojson('malha_brasileira.json')

psr = carrega_parquet('PSR.parquet')[['NM_RAZAO_SOCIAL', 'NM_CULTURA_GLOBAL', 'PE_TAXA', 'NR_PRODUTIVIDADE_SEGURADA', 'uf', 'ibge', 'ano', 'descricao_tipologia', 'municipio', 'NR_APOLICE', 'VL_PREMIO_LIQUIDO', 'VL_SUBVENCAO_FEDERAL', 'VALOR_INDENIZAÇÃO']]


estados = {
    'Acre': 'AC',
    'Alagoas': 'AL',
    'Amazonas': 'AM',
    'Amapá': 'AP',
    'Bahia': 'BA',
    'Ceará': 'CE',
    'Distrito Federal': 'DF',
    'Espírito Santo': 'ES',
    'Goiás': 'GO',
    'Maranhão': 'MA',
    'Minas Gerais': 'MG',
    'Mato Grosso do Sul': 'MS',
    'Mato Grosso': 'MT',
    'Pará': 'PA',
    'Paraíba': 'PB',
    'Pernambuco': 'PE',
    'Piauí': 'PI',
    'Paraná': 'PR',
    'Rio de Janeiro': 'RJ',
    'Rio Grande do Norte': 'RN',
    'Rondônia': 'RO',
    'Roraima': 'RR',
    'Rio Grande do Sul': 'RS',
    'Santa Catarina': 'SC',
    'Sergipe': 'SE',
    'São Paulo': 'SP',
    'Tocantins': 'TO'
}


# estados = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
anos = np.arange(1991, 2023)
anos_latam = np.arange(2000, 2024)
anos_psr = np.arange(2006, 2022)
mapa_de_cores = {
    'Estiagem e Seca': '#EECA3B',
    'Incêndio Florestal': '#E45756',
    'Onda de Frio': '#72B7B2',
    'Onda de Calor e Baixa Umidade': '#F58518',
    'Enxurradas': '#B279A2',
    'Inundações': '#0099C6',
    'Alagamentos': '#72B7B2',
    'Movimento de Massa': '#9D755D',
    'Chuvas Intensas': '#4C78A8',
    'Vendavais e Ciclones': '#54A24B',
    'Granizo': 'rgb(102, 102, 102)',
    'Tornado': '#4C78A8',
    'Onda de Frio': '#72B7B2',
    'Doenças infecciosas': '#54A24B',
    'Erosão': '#9D755D',
    'Outros': '#FF9DA6',
    'Rompimento/Colapso de barragens': 'rgb(102, 102, 102)',
    'Sem Dados': '#BAB0AC'
}
cores_risco = {
    'Muito Alto': '#E45756',
    'Alto': '#F58518',
    'Moderado': '#EECA3B',
    'Baixo': '#72B7B2',
    'Muito Baixo': '#4C78A8'
}
cores_segurado = {
    'nao segurado': '#EECA3B',
    'segurado sem sinistro': '#54A24B',
    'segurado com sinistro': '#E45756'
}
desastres = {
    'Hidrológico': ['Alagamentos', 'Chuvas Intensas', 'Enxurradas', 'Inundações', 'Movimento de Massa'],
    'Climatológico': ['Estiagem e Seca', 'Incêndio Florestal', 'Onda de Calor e Baixa Umidade', 'Onda de Frio'],
    'Meteorológico': ['Granizo', 'Onda de Frio', 'Tornado', 'Vendavais e Ciclones'],
    'Outros': ['Doenças infecciosas', 'Erosão', 'Onda de Calor e Baixa Umidade', 'Outros', 'Rompimento/Colapso de barragens']
}
idx_select = {
    'Climatológico': 0,
    'Hidrológico': 2,
    'Meteorológico': 3,
    'Outros': 1
}

idx_select_br = {
    'Climatológico': 0,
    'Hidrológico': 3,
    'Meteorológico': 3,
    'Outros': 1
}
seg = {
    'BRASILSEG COMPANHIA DE SEGUROS': 'Brasilseg', 
    'Mapfre Seguros Gerais S.A.': 'MAPFRE Seguros',
    'Essor Seguros S.A.': 'Essor Seguros',
    'Swiss Re Corporate Solutions Brasil S.A.': 'Swiss Re',
    'Nobre Seguradora do Brasil S.A': 'Nobre Seguradora',
    'Allianz Seguros S.A': 'Allianz Seguros',
    'Sancor Seguros do Brasil S.A.': 'Sancor Seguros',
    'FairFax Brasil Seguros Corporativos S/A': 'Fairfax Seguros',
    'Newe Seguros S.A': 'Newe Seguros',
    'Tokio Marine Seguradora S.A.': 'Tokio Marine Seguradora',
    'Porto Seguro Companhia de Seguros Gerais': 'Porto Seguro',
    'Too Seguros S.A.': 'Too Seguros',
    'Aliança do Brasil Seguros S/A.': 'Aliança do Brasil Seguros',
    'Sompo Seguros S/A': 'Sompo Seguros',
    'Companhia Excelsior de Seguros': 'Seguros Excelsior',
    'EZZE Seguros S.A.': 'EZZE Seguros',
    'Itaú XL Seguros Corporativos S.A': 'Itaú XL Seguros',
}
psr.NM_RAZAO_SOCIAL = psr.NM_RAZAO_SOCIAL.map(seg)
psr.PE_TAXA = psr.PE_TAXA * 100

# COLUNAS
tabs = st.tabs(['UF do Brasil', 'Agro', 'América Latina', 'Créditos'])

with tabs[0]:
    col_mapa, col_dados = st.columns([1, 1], gap='large')
    select1, select2 = col_dados.columns([1, 1])



    # SELECTBOX
    uf_selectbox = select1.selectbox('Selecione o estado', list(estados.keys()), index=23)
    uf_selecionado = estados[uf_selectbox]
    grupo_desastre_selecionado = select2.selectbox('Selecione o grupo de desastre', list(desastres.keys()), index=0)
    coord_municipio = select1.selectbox('Encontrar município',['-'] + dados_merge.iloc[:-45].query("abbrev_state == @uf_selecionado").name_muni.unique().tolist(), index=0)
    ano_inicial, ano_final = select2.select_slider('Selecione o período', anos, value=(anos[0], anos[-1]))



    # BUBBLE PLOT
    atlas_year = dados_atlas.query("grupo_de_desastre == @grupo_desastre_selecionado & uf == @uf_selecionado & ano >= @ano_inicial & ano <= @ano_final").groupby(['ano', 'descricao_tipologia'], as_index=False).size().rename(columns={'size': 'ocorrencias'})



    fig_grupo_desastre = px.scatter(atlas_year, x="ano", y='descricao_tipologia', size='ocorrencias', 
        color='descricao_tipologia', size_max=50, color_discrete_map=mapa_de_cores,
        labels={
            "ano": "Ano", 
            "descricao_tipologia": "Desastre"
        }
    )
    fig_grupo_desastre.update_layout(showlegend=False, legend_orientation='h', margin={"r":0,"t":0,"l":0,"b":0})
    fig_grupo_desastre.update_xaxes(showgrid=True)
    # col_dados.caption('Quanto maior o círculo, maior o número de ocorrências do desastre')
    col_dados.plotly_chart(fig_grupo_desastre)



    # selecionando estado
    tipologia_selecionada = col_dados.selectbox('Selecione a tipologia do desastre', desastres[grupo_desastre_selecionado], index=idx_select[grupo_desastre_selecionado], key='tipol')



    # MALHA
    malha_mun_estados = carrega_malha(uf=uf_selecionado)
    zoom_uf = 5
    if coord_municipio == '-':
        lat, lon = coord_uf.query("abbrev_state == @uf_selecionado")[['lat', 'lon']].values[0]
    else:
        cod_muni = dados_merge.loc[dados_merge.name_muni == coord_municipio, 'code_muni'].values[0]
        lat, lon = coord_muni.query("codarea == @cod_muni")[['lat', 'lon']].values[0]
        zoom_uf = 10



    # MAPA DE DESASTRES COMUNS
    tipologias_mais_comuns_por_muni = dados_atlas.query("grupo_de_desastre == @grupo_desastre_selecionado & uf == @uf_selecionado & ano >= @ano_inicial & ano <= @ano_final").groupby(['ibge', 'descricao_tipologia'], as_index=False).size().sort_values('size', ascending=False).drop_duplicates(subset='ibge', keep='first').rename(columns={'size': 'ocorrencias', 'descricao_tipologia': 'desastre_mais_comum'})
    merge_muni_2 = dados_merge.query("abbrev_state == @uf_selecionado").groupby(['code_muni', 'name_muni'], as_index=False).size().drop('size', axis=1)
    tipol_merge = merge_muni_2.merge(tipologias_mais_comuns_por_muni, how='left', left_on='code_muni', right_on='ibge').drop('ibge', axis=1)
    tipol_merge.loc[np.isnan(tipol_merge["ocorrencias"]), 'ocorrencias'] = 0
    tipol_merge.desastre_mais_comum = tipol_merge.desastre_mais_comum.fillna('Sem Dados')
    col_mapa.subheader(f'Desastre mais comum por Município ({ano_inicial} - {ano_final})')
    col_mapa.plotly_chart(cria_mapa(tipol_merge, malha_mun_estados, locais='code_muni', cor='desastre_mais_comum', lista_cores=mapa_de_cores, nome_hover='name_muni', dados_hover=['desastre_mais_comum', 'ocorrencias'], zoom=zoom_uf, lat=lat, lon=lon, titulo_legenda='Desastre mais comum'), use_container_width=True)



    # QUERY
    dados_atlas_query = dados_atlas.query("descricao_tipologia == @tipologia_selecionada & uf == @uf_selecionado & ano >= @ano_inicial & ano <= @ano_final")



    # MAPA RISCO
    ocorrencias = dados_atlas_query.groupby(['ibge', 'municipio'], as_index=False).size().rename(columns={'size': 'ocorrencias'}).sort_values('ocorrencias', ascending=False).drop_duplicates(subset='ibge', keep='first')
    merge_muni = dados_merge.query("abbrev_state == @uf_selecionado").groupby(['code_muni', 'name_muni', 'AREA_KM2'], as_index=False).size().drop('size', axis=1).drop_duplicates(subset='code_muni', keep='first')
    ocorrencias_merge = merge_muni.merge(ocorrencias, how='left', left_on='code_muni', right_on='ibge')
    ocorrencias_merge.loc[np.isnan(ocorrencias_merge["ocorrencias"]), 'ocorrencias'] = 0

    classificacao_ocorrencias = classifica_risco(ocorrencias_merge, 'ocorrencias')  # mudadr classficador
    fig_mapa = cria_mapa(classificacao_ocorrencias, malha_mun_estados, locais='code_muni', cor='risco', lista_cores=cores_risco, dados_hover='ocorrencias', nome_hover='name_muni', lat=lat, lon=lon, zoom=5, titulo_legenda=f'Risco de {tipologia_selecionada}')
    # fig_mapa = cria_mapa(classificacao_ocorrencias, malha_mun_estados, locais='code_muni', cor='ocorrencias', tons=list(cores_risco.values()), dados_hover='ocorrencias', nome_hover='name_muni', lat=lat, lon=lon, zoom=5, titulo_legenda=f'Risco de {tipologia_selecionada}')
    col_mapa.divider()
    col_mapa.subheader(f'Risco de {tipologia_selecionada} em {uf_selecionado} ({ano_inicial} - {ano_final})')
    col_mapa.plotly_chart(fig_mapa, use_container_width=True)



    # MÉTRICAS
    met1, met2 = col_dados.columns([1, 1])
    met3, met4 = col_dados.columns([1, 1])

    met1.metric('Total de Ocorrências', len(dados_atlas_query))
    met2.metric('Média de Ocorrências por Ano', dados_atlas_query.groupby('ano').size().mean().astype(int))
    muni_ocorr = math.ceil(len(classificacao_ocorrencias.query("ocorrencias > 0")) / len(classificacao_ocorrencias) * 100)
    met3.metric('% dos Municípios com no *mínimo* Uma Ocorrência', f'{muni_ocorr}%')
    area_risco = math.ceil(classificacao_ocorrencias.loc[classificacao_ocorrencias.query("risco == 'Muito Alto' | risco == 'Alto'").index, "AREA_KM2"].sum() / classificacao_ocorrencias.AREA_KM2.sum() * 100)
    met4.metric('% de Área Classificada como Risco *Alto* e *Muito Alto*', f'{area_risco}%')



    # DATAFRAME E DOWNLOAD
    tabela = ocorrencias.copy().reset_index(drop=True).sort_values('ocorrencias', ascending=False).rename(columns={'ibge': 'codigo_municipal'})
    tabela['ocorrencias_por_ano'] = tabela.ocorrencias / (ano_final - ano_inicial + 1)
    tabela_merge = tabela.merge(pop_pib, how='left', left_on='codigo_municipal', right_on='code_muni').drop('code_muni', axis=1)

    expander = col_dados.expander(f'Municípios com o maior risco de *{tipologia_selecionada}* em {uf_selecionado}', expanded=True)
    expander.dataframe(tabela_merge.head(), hide_index=True,
                       column_config={
                            'codigo_municipal': None,
                            'municipio': st.column_config.TextColumn('Município'),
                            'ocorrencias': st.column_config.TextColumn('Total ocorrências'),
                            'pib_per_capita': st.column_config.NumberColumn(
                                'PIB per Capita',
                                format="R$ %.2f",
                            ),
                            'populacao': st.column_config.NumberColumn('Pop.', format='%d'),
                            'ocorrencias_por_ano': st.column_config.NumberColumn('Média ocorrências/ano', format='%.1f')
                        })
    
    col_dados.download_button('Baixar tabela', tabela_merge.to_csv(sep=';', index=False), file_name=f'ocorrencias_{uf_selecionado}.csv', mime='text/csv', use_container_width=True)



    # psrQ1_muni = psr.query("uf == @uf_selecionado")
    # psrQ2_muni = psrQ1_muni.query("descricao_tipologia == @tipologia_selecionada & ano >= @ano_inicial & ano <= @ano_final")
    # psrG_muni = psrQ2_muni.groupby('municipio').agg({
    #     'descricao_tipologia': 'count',
    #     'NM_CULTURA_GLOBAL': lambda x: x.mode().iloc[0],
    #     'PE_TAXA': 'mean',
    #     'NR_PRODUTIVIDADE_SEGURADA': 'mean',
    #     'NM_RAZAO_SOCIAL': lambda x: x.mode().iloc[0],
    # }).reset_index()
    # psrApol_muni = psrQ2_muni.groupby(['municipio'], as_index=False).size().merge(psrQ1_muni.groupby(['municipio'], as_index=False)['NR_APOLICE'].nunique(), how='left', on='municipio')
    # psrG_muni['apolices'] = psrApol_muni['NR_APOLICE']
    # psrG_muni['sin/apol'] = (psrApol_muni['size'] / psrApol_muni['NR_APOLICE']) * 100



    # # PSR RISCO
    # if tipologia_selecionada in psr.descricao_tipologia.unique().tolist():
    #     col_dados.subheader(f'Dados da PSR para *{tipologia_selecionada}* em {uf_selecionado} ({ano_inicial} - {ano_final})')



    #     # PSR MÉTRICAS
    #     met_psr1, met_psr2 = col_dados.columns([1, 1])
    #     met_psr1.metric('Total de Sinistros', len(psrQ2_muni))
    #     met_psr2.metric('Total de Apólices', psrApol_muni.NR_APOLICE.sum())



    #     psrQ1 = psr.query("uf == @uf_selecionado")
    #     psrQ2 = psrQ1.query("descricao_tipologia == @tipologia_selecionada & ano >= @ano_inicial & ano <= @ano_final")
    #     print(psrQ2.NM_CULTURA_GLOBAL)
    #     merge_muni_psr = dados_merge.query("abbrev_state == @uf_selecionado")
    #     sin = psrQ2.groupby(['ibge'], as_index=False).size()
    #     sin_merge = merge_muni_psr.merge(sin, how='left', left_on='code_muni', right_on='ibge').rename(columns={'size': 'sinistros'})
    #     sin_merge.sinistros = sin_merge.sinistros.fillna(0)
    #     sin_risco = classifica_risco(sin_merge, 'sinistros')
    #     sin_fig = cria_mapa(sin_risco, malha_mun_estados, locais='code_muni', cor='risco', lista_cores=cores_risco, dados_hover='sinistros', nome_hover='name_muni', lat=lat, lon=lon, zoom=5, titulo_legenda=f'Risco de Sinistros de {tipologia_selecionada}')
    #     col_mapa.divider()
    #     col_mapa.subheader(f'Sinistros de {tipologia_selecionada} nos dados da PSR por Município ({ano_inicial} - {ano_final})')
    #     col_mapa.plotly_chart(sin_fig, use_container_width=True)



    #     psrPie = psrQ2.NM_CULTURA_GLOBAL.value_counts()
    #     figpie = px.pie(
    #         psrPie,
    #         values='count',
    #         names=psrPie.index,
    #         title=f'Culturas mais afetadas por {tipologia_selecionada} nos dados da PSR'
    #     )
    #     figpie.update_layout(
    #         legend=dict(font=dict(size=16)),
    #         legend_title=dict(font=dict(size=14), text='Culturas')
    #     )
    #     col_dados.plotly_chart(figpie, use_container_width=True)

    
    
    # # PSR
    # # psr.groupby(['municipio'], as_index=False).size()
    # if tipologia_selecionada in psr.descricao_tipologia.unique().tolist():
    #     st.divider()
    #     st.subheader(f'Sinistros de {tipologia_selecionada} nos dados da PSR por Município (2006 - 2021)')
    #     st.dataframe(
    #         psrG_muni.sort_values('descricao_tipologia', ascending=False).head(10),
    #         hide_index=True, 
    #         column_config={
    #             'municipio': st.column_config.TextColumn('Município'),
    #             'descricao_tipologia': st.column_config.TextColumn('Total Sinistros'),
    #             'apolices': st.column_config.TextColumn('Total Apólices'),
    #             'NM_CULTURA_GLOBAL': st.column_config.TextColumn('Cultura mais Comum'),
    #             'NM_RAZAO_SOCIAL': st.column_config.TextColumn('Seguradora mais Comum'),
    #             'NR_PRODUTIVIDADE_SEGURADA': st.column_config.NumberColumn(
    #                 'Média Prod. Segurada',
    #                 format="%.2f",
    #             ),
    #             'PE_TAXA': st.column_config.NumberColumn(
    #                 'Média Taxa de Prêmio',
    #                 format="%.2f%%",
    #             ),
    #             'sin/apol': st.column_config.NumberColumn(
    #                 'Perc. de Sinistros por Apólice',
    #                 format="%.2f%%",
    #             )
    #         },
    #         use_container_width=True
    #     )



    # LINEPLOT
    line_query = dados_atlas.iloc[:62273].query("descricao_tipologia == @tipologia_selecionada & uf == @uf_selecionado & ano >= 2000")
    cols_danos = ['agricultura', 'pecuaria', 'industria']  # 'total_danos_materiais'
    soma_danos = line_query.groupby(['ano'], as_index=False)[cols_danos].sum()
    st.subheader(f'Danos causados por *{tipologia_selecionada}* em *{uf_selecionado} de 2000 a 2022*')

    fig_line = px.line(
        soma_danos, 'ano', cols_danos, markers=True, 
        labels={'value': 'Valor', 'variable': 'Setor', 'ano': 'Ano'}, 
        line_shape='spline'
    )
    fig_line.update_layout(
    legend=dict(orientation="v",
        font=dict(size=16))
    )
    st.plotly_chart(fig_line, use_container_width=True)      



    # HEATMAPS
    aba_hm1, aba_hm2 = st.tabs(['Ocorrências por Grupo de Desastre', 'Ocorrências por Estado'])
    cls_scales = {
        'Climatológico': 'OrRd',
        'Hidrológico': 'PuBu',
        'Meteorológico': 'Tempo',
        'Outros': 'Brwnyl'
    }

    with aba_hm2:
        heatmap_query = dados_atlas.iloc[:62273].query("descricao_tipologia == @tipologia_selecionada & ano >= 2000")
        pivot_hm = heatmap_query.pivot_table(index='ano', columns='uf', aggfunc='size', fill_value=0)
        pivot_hm = pivot_hm.reindex(columns=dados_atlas.uf.unique()[:-1], fill_value=0)
        pivot_hm = pivot_hm.reindex(index=anos, fill_value=0).transpose()
        fig_hm = px.imshow(
            pivot_hm,
            labels=dict(x="Ano", y="Estado (UF)", color="Total ocorrências"),
            x=pivot_hm.columns,
            y=pivot_hm.index,
            color_continuous_scale=cls_scales[grupo_desastre_selecionado],
        )
        fig_hm.update_layout(
            yaxis_nticks=len(pivot_hm),
            height=700
        )
        st.subheader(f'Ocorrências de *{tipologia_selecionada}* por estado de 1991 a 2022')
        st.plotly_chart(fig_hm, use_container_width=True)
    with aba_hm1:
        heatmap_query2 = dados_atlas.iloc[:62273].query("grupo_de_desastre == @grupo_desastre_selecionado & uf == @uf_selecionado & ano >= 2000")
        pivot_hm2 = heatmap_query2.pivot_table(index='ano', columns='descricao_tipologia', aggfunc='size', fill_value=0)
        pivot_hm2 = pivot_hm2.reindex(index=anos, fill_value=0).transpose()
        fig_hm2 = px.imshow(
            pivot_hm2,
            labels=dict(x="Ano", y="Desastre", color="Total ocorrências"),
            x=pivot_hm2.columns,
            y=pivot_hm2.index,
            color_continuous_scale=cls_scales[grupo_desastre_selecionado],
        )
        fig_hm2.update_layout(
            yaxis_nticks=len(pivot_hm2),
        )
        st.subheader(f'Ocorrências do grupo de desastre *{grupo_desastre_selecionado} em {uf_selecionado}* de 1991 a 2022')
        st.plotly_chart(fig_hm2, use_container_width=True)



with tabs[1]:
    col_config, col_mapa_agro, col_metrics = st.columns([1, 3, 1], gap='large')

    # CONFIG
    uf_psr = estados[col_config.selectbox('Estado (UF)', estados.keys(), index=23, key='uf_psr')]
    tipologia_selecionada_psr = col_config.selectbox('Tipologia', psr.descricao_tipologia.unique()[1:], index=2, key='tipol_psr')
    ano_inicial_psr, ano_final_psr = col_config.select_slider('Período', anos_psr, value=(anos_psr[0], anos_psr[-1]), key='periodo_psr')



    # AGRO
    psrQ1 = psr.query("uf == @uf_psr & ano >= @ano_inicial_psr & ano <= @ano_final_psr")
    psrQ2 = psrQ1.query("descricao_tipologia == @tipologia_selecionada_psr")
  
    merge_muni_psr = dados_merge.iloc[:-45].query("abbrev_state == @uf_psr")
    sin = psrQ2.groupby(['ibge'], as_index=False).size()
    sin_merge = merge_muni_psr.merge(sin, how='left', left_on='code_muni', right_on='ibge').rename(columns={'size': 'sinistros'})
    sin_merge.sinistros = sin_merge.sinistros.fillna(0)
    sin_merge.ibge = sin_merge.ibge.fillna('-')
    sin_segurado = classifica_segurado(psrQ1, sin_merge)
    # sin_risco = classifica_risco(sin_merge, 'sinistros')

    malha_psr = carrega_malha(uf=uf_psr)
    lat_psr, lon_psr = coord_uf.query("abbrev_state == @uf_psr")[['lat', 'lon']].values[0]
    sin_fig = cria_mapa(sin_segurado, malha_psr, locais='code_muni', cor='seg', lista_cores=cores_segurado, dados_hover='sinistros', nome_hover='name_muni', lat=lat_psr, lon=lon_psr, zoom=6, titulo_legenda=f'Sinistros de {tipologia_selecionada_psr}')

    col_mapa_agro.subheader(f'Sinistros de {tipologia_selecionada_psr} por Município ({ano_inicial_psr} - {ano_final_psr})')
    col_mapa_agro.plotly_chart(sin_fig, use_container_width=True)


    lr = psrQ1.groupby(['uf'], as_index=False)[['VL_PREMIO_LIQUIDO', 'VL_SUBVENCAO_FEDERAL', 'VALOR_INDENIZAÇÃO']].sum()
    lr['loss_ratio'] = lr.VALOR_INDENIZAÇÃO / (lr.VL_PREMIO_LIQUIDO + lr.VL_SUBVENCAO_FEDERAL)


    # METRICS
    col_metrics.metric('Total de Apólices', psrQ1.NR_APOLICE.nunique())
    col_metrics.metric(f'Sinistros de {tipologia_selecionada_psr}', len(psrQ2))
    lr_metric = f'{lr.loss_ratio.multiply(100).astype(int).values[0]}%'
    col_metrics.metric('Loss Ratio', lr_metric)



with tabs[2]:
    col_mapa_br, col_dados_br = st.columns([1, 1], gap='large')
 


    # SELECTBOX
    grupo_desastre_selecionado_br = col_dados_br.selectbox('Selecione o grupo de desastre', list(desastres.keys()), index=0, key='gp_desastre_br')
    ano_inicial_br, ano_final_br = col_dados_br.select_slider('Selecione o período', anos_latam, value=(anos_latam[0], anos_latam[-1]), key='periodo_br')



    # QUERY
    dados_atlas_query_br_1 = dados_atlas.query("grupo_de_desastre == @grupo_desastre_selecionado_br & ano >= @ano_inicial_br & ano <= @ano_final_br")
    


    # BUBBLE PLOT
    atlas_year_br = dados_atlas_query_br_1.groupby(['ano', 'descricao_tipologia'], as_index=False).size().rename(columns={'size': 'ocorrencias'})



    fig_grupo_desastre_br = px.scatter(atlas_year_br, x="ano", y='descricao_tipologia', size='ocorrencias', 
        color='descricao_tipologia', size_max=50, color_discrete_map=mapa_de_cores,
        labels={
            "ano": "Ano", 
            "descricao_tipologia": "Desastre"
        }
    )
    fig_grupo_desastre_br.update_layout(showlegend=False, legend_orientation='h', margin={"r":0,"t":0,"l":0,"b":0})
    fig_grupo_desastre_br.update_xaxes(showgrid=True)
    # col_dados_br.caption('Quanto maior o círculo, maior o número de ocorrências do desastre')
    col_dados_br.plotly_chart(fig_grupo_desastre_br)



    # selecionando estado
    col_pais, col_desastre = col_dados_br.columns([1, 1])

    pais_selecionado = col_pais.selectbox('Selecione o país', sorted(dados_merge.iloc[-45:].name_state.unique()), index=7, key='pais_br')
    iso = dados_merge.loc[dados_merge.name_state == pais_selecionado, 'code_state'].values[0]
    malha_pais_selecionado = malha_brasil if iso == 'BRA' else filtra_geojson(malha_america, iso)
    
    tipologia_selecionada_br = col_desastre.selectbox('Selecione a tipologia do desastre', desastres[grupo_desastre_selecionado_br], index=idx_select_br[grupo_desastre_selecionado], key='tipol_br')



    # MAPA DE DESASTRES COMUNS
    tipologias_mais_comuns_por_estado = dados_atlas_query_br_1.groupby(['pais', 'descricao_tipologia'], as_index=False).size().sort_values('size', ascending=False).drop_duplicates(subset='pais', keep='first').rename(columns={'size': 'ocorrencias', 'descricao_tipologia': 'desastre_mais_comum'})
    tipol_br = dados_merge.groupby(['code_state', 'name_state'], as_index=False).size().drop('size', axis=1)
    tipol_merge_br = tipol_br.merge(tipologias_mais_comuns_por_estado, how='left', left_on='name_state', right_on='pais').drop('pais', axis=1)
    tipol_merge_br.loc[np.isnan(tipol_merge_br['ocorrencias']), 'ocorrencias'] = 0
    tipol_merge_br.desastre_mais_comum = tipol_merge_br.desastre_mais_comum.fillna('Sem Dados')
    col_mapa_br.subheader(f'Desastres mais comuns por País ({ano_inicial_br} - {ano_final_br})')
    col_mapa_br.plotly_chart(cria_mapa(tipol_merge_br, malha_america, locais='code_state', cor='desastre_mais_comum', lista_cores=mapa_de_cores, nome_hover='name_state', dados_hover=['desastre_mais_comum', 'ocorrencias'], zoom=1, titulo_legenda='Desastre mais comum'), use_container_width=True)



    # QUERY
    dados_atlas_query_br_2 = dados_atlas_query_br_1.query("descricao_tipologia == @tipologia_selecionada_br")



    # MAPA RISCO
    col_mapa_br.divider()  
    col_mapa_br.subheader(f'{pais_selecionado}: Risco de {tipologia_selecionada_br} ({ano_inicial_br} - {ano_final_br})')

    ocorrencias_br = dados_atlas_query_br_2.groupby(['cod_uf', 'pais'], as_index=False).size().rename(columns={'size': 'ocorrencias'})

    merge_ufs = dados_merge.iloc[:-45].groupby(['code_state', 'name_state'], as_index=False).size().drop('size', axis=1)
    merge_paises = dados_merge.iloc[-45:].drop(['code_muni', 'name_muni'], axis=1)
    merge_escolhido = merge_ufs if iso == 'BRA' else merge_paises
    ocorrencias_merge_br = merge_escolhido.merge(ocorrencias_br, how='left', left_on='code_state', right_on='cod_uf')
    ocorrencias_merge_br.loc[np.isnan(ocorrencias_merge_br["ocorrencias"]), 'ocorrencias'] = 0
    classificacao_ocorrencias_br = classifica_risco(ocorrencias_merge_br, 'ocorrencias')

    fig_mapa_br = cria_mapa(classificacao_ocorrencias_br, malha_pais_selecionado, locais='code_state', cor='risco', lista_cores=cores_risco, dados_hover='ocorrencias', nome_hover='name_state', titulo_legenda=f'Risco de {tipologia_selecionada_br}', zoom=1, featureid='properties.codarea')

    coord_pais = coord_latam.query("cod_uf == @iso & ano >= @ano_inicial_br & ano <= @ano_final_br & descricao_tipologia == @tipologia_selecionada_br")
    if iso != 'BRA':
        fig_mapa_br.add_trace(
            px.scatter_mapbox(
                lat=coord_pais['latitude'],
                lon=coord_pais['longitude'],
                hover_name=coord_pais['local']
            ).data[0]
        )
        fig_mapa_br.update_traces(marker=dict(size=12, color='#222A2A', selector=dict(mode='markers')), selector=dict(mode='markers'))



    # fig_mapa_br = cria_mapa(classificacao_ocorrencias_br, malha_america, locais='code_state', cor='ocorrencias', tons=list(cores_risco.values()), dados_hover='ocorrencias', nome_hover='name_state', titulo_legenda=f'Risco de {tipologia_selecionada_br}')

    # scatter geo
    # coord_selecionada = coord_latam.query("cod_uf == @iso & ano >= @ano_inicial_br & ano <= @ano_final_br & descricao_tipologia == @tipologia_selecionada_br")
    # print(coord_selecionada.head())
    # col_mapa_br.map(coord_selecionada, latitude='latitude', longitude='longitude', size=1000, zoom=3, use_container_width=True)

    col_mapa_br.plotly_chart(fig_mapa_br, use_container_width=True)



    # if iso =='BRA' and (tipologia_selecionada_br in psr.descricao_tipologia.unique().tolist()):
    #     psrQBR = psr.query("descricao_tipologia == @tipologia_selecionada_br & ano >= @ano_inicial_br & ano <= @ano_final_br")
    #     merge_BR_psr = dados_merge.iloc[:-45].groupby(['abbrev_state', 'code_state', 'name_state'], as_index=False).size().drop('size', axis=1)
    #     sinBR = psrQBR.groupby(['uf'], as_index=False).size()
    #     sin_mergeBR = merge_BR_psr.merge(sinBR, how='left', left_on='abbrev_state', right_on='uf').rename(columns={'size': 'sinistros'})
    #     sin_mergeBR.sinistros = sin_mergeBR.sinistros.fillna(0)
    #     sin_riscoBR = classifica_risco(sin_mergeBR, 'sinistros')
    #     sin_figBR = cria_mapa(sin_riscoBR, malha_brasil, locais='code_state', cor='risco', lista_cores=cores_risco, dados_hover='sinistros', nome_hover='name_state', lat=-15.78, lon=-47.92, zoom=3, titulo_legenda=f'Risco de Sinistros de {tipologia_selecionada_br}')
    #     col_mapa_br.subheader(f'Sinistros de {tipologia_selecionada_br} nos dados da PSR por Município')
    #     col_mapa_br.plotly_chart(sin_figBR, use_container_width=True)



    # DADOS
    dados_tabela = dados_atlas_query_br_1.query("descricao_tipologia == @tipologia_selecionada_br").groupby(['pais'], as_index=False).size().rename(columns={'size': 'ocorrencias'})
    tabela_br = dados_tabela.copy().reset_index(drop=True).sort_values('ocorrencias', ascending=False)
    tabela_br['ocorrencias_por_ano'] = round(tabela_br.ocorrencias.div(ano_final_br - ano_inicial_br + 1), 1)
  
    tabela_merge_br = tabela_br.merge(pop_pib_uf, how='right', left_on='pais', right_on='pais')
    tabela_merge_br.loc[np.isnan(tabela_merge_br["ocorrencias"]), 'ocorrencias'] = 0
    tabela_merge_br.loc[np.isnan(tabela_merge_br["ocorrencias_por_ano"]), 'ocorrencias_por_ano'] = 0.0
    tabela_merge_br = tabela_merge_br.sort_values('ocorrencias', ascending=False)
    tabela_merge_br.loc[tabela_merge_br.query("cod_uf == 'VEN'").index, 'pais'] = 'Venezuela'



    # MÉTRICAS
    met1_br, met2_br = col_dados_br.columns([1, 1])
    met1_br.metric('Total de ocorrências', tabela_merge_br.query("pais == @pais_selecionado")['ocorrencias'])
    met2_br.metric('Média de ocorrências por ano', tabela_merge_br.query("pais == @pais_selecionado")['ocorrencias_por_ano'])

    

    # DATAFRAME E DOWNLOAD
    expander_br = col_dados_br.expander(f'Países com o maior risco de *{tipologia_selecionada_br}* na América Latina', expanded=True)
    expander_br.dataframe(tabela_merge_br.head(), hide_index=True, 
                          column_config={
                            'pais': st.column_config.TextColumn('País'),
                            'ocorrencias': st.column_config.TextColumn('Total ocorrências'),
                            'pib_per_capita': st.column_config.NumberColumn(
                                'PIB per Capita',
                                format="R$ %.2f",
                            ),
                            'populacao': st.column_config.NumberColumn('Pop.', format='%d'),
                            'ocorrencias_por_ano': st.column_config.NumberColumn('Média ocorrências/ano', format='%.1f')
                        })

    col_dados_br.download_button('Baixar tabela', tabela_merge_br.to_csv(sep=';', index=False), file_name=f'{tipologia_selecionada_br.replace(" ", "_").lower()}_americalatina.csv', mime='text/csv', use_container_width=True)



    # PSR
    # psr.groupby(['uf'], as_index=False).size()
    # if tipologia_selecionada_br in psr.descricao_tipologia.unique().tolist():
    #     psrQ = psr.query("descricao_tipologia == @tipologia_selecionada_br & ano >= @ano_inicial_br & ano <= @ano_final_br")
    #     psrG = psrQ.groupby('uf').agg({
    #         'descricao_tipologia': 'count',
    #         'NM_CULTURA_GLOBAL': lambda x: x.mode().iloc[0],
    #         'PE_TAXA': 'mean',
    #         'NR_PRODUTIVIDADE_SEGURADA': 'mean',
    #         'NM_RAZAO_SOCIAL': lambda x: x.mode().iloc[0],
    #     }).reset_index()
    #     psrApol = psrQ.groupby(['uf'], as_index=False).size().merge(psr.groupby(['uf'], as_index=False)['NR_APOLICE'].nunique(), how='left', on='uf')
    #     psrG['apolices'] = psrApol['NR_APOLICE']
    #     psrG['sin/apol'] = (psrApol['size'] / psrApol['NR_APOLICE']) * 100
    #     st.divider()
    #     st.subheader(f'Sinistros de {tipologia_selecionada_br} nos dados da PSR por UF (2006 - 2021)')
    #     st.dataframe(
    #         psrG,
    #         hide_index=True, 
    #         column_config={
    #             'uf': st.column_config.TextColumn('Estado (UF)'),
    #             'descricao_tipologia': st.column_config.TextColumn('Total Sinistros'),
    #             'apolices': st.column_config.TextColumn('Total Apólices'),
    #             'NM_CULTURA_GLOBAL': st.column_config.TextColumn('Cultura mais Comum'),
    #             'NM_RAZAO_SOCIAL': st.column_config.TextColumn('Seguradora mais Comum'),
    #             'NR_PRODUTIVIDADE_SEGURADA': st.column_config.NumberColumn(
    #                 'Média Prod. Segurada',
    #                 format="%.2f",
    #             ),
    #             'PE_TAXA': st.column_config.NumberColumn(
    #                 'Média Taxa de Prêmio',
    #                 format="%.2f%%",
    #             ),
    #             'sin/apol': st.column_config.NumberColumn(
    #                 'Perc. de Sinistros por Apólice',
    #                 format="%.2f%%",
    #             )
    #         },
    #         use_container_width=True
    #     )



    aba_hm1_br, aba_hm2_br = st.tabs(['Ocorrências por Grupo de Desastre', 'Ocorrências por País'])

    with aba_hm2_br:
        heatmap_query_br = dados_atlas.iloc[62273:].query("descricao_tipologia == @tipologia_selecionada & ano >= 2000")
        pivot_hm_br = heatmap_query_br.pivot_table(index='ano', columns='pais', aggfunc='size', fill_value=0)
        # pivot_hm_br = pivot_hm_br.reindex(columns=dados_atlas.pais.unique(), fill_value=0)
        pivot_hm_br = pivot_hm_br.reindex(index=anos_latam, fill_value=0).transpose()
        fig_hm_br = px.imshow(
            pivot_hm_br,
            labels=dict(x="Ano", y="País", color="Total ocorrências"),
            x=pivot_hm_br.columns,
            y=pivot_hm_br.index,
            color_continuous_scale=cls_scales[grupo_desastre_selecionado_br],
        )
        fig_hm_br.update_layout(
            yaxis_nticks=len(pivot_hm_br),
            height=700
        )
        st.subheader(f'Ocorrências de *{tipologia_selecionada_br}* por País de 2000 a 2023')
        st.caption('Países sem ocorrências não aparecem no gráfico')
        st.plotly_chart(fig_hm_br, use_container_width=True)
    with aba_hm1_br:
 
        heatmap_query2_br = dados_atlas.query("grupo_de_desastre == @grupo_desastre_selecionado & pais == @pais_selecionado & ano >= 2000")
        pivot_hm2_br = heatmap_query2_br.pivot_table(index='ano', columns='descricao_tipologia', aggfunc='size', fill_value=0)
        pivot_hm2_br = pivot_hm2_br.reindex(index=anos_latam, fill_value=0).transpose()
        fig_hm2_br = px.imshow(
            pivot_hm2_br,
            labels=dict(x="Ano", y="Desastre", color="Total ocorrências"),
            x=pivot_hm2_br.columns,
            y=pivot_hm2_br.index,
            color_continuous_scale=cls_scales[grupo_desastre_selecionado_br],
        )
        fig_hm2_br.update_layout(
            yaxis_nticks=len(pivot_hm2_br),
        )
        st.subheader(f'{pais_selecionado}: Ocorrências do grupo de desastre *{grupo_desastre_selecionado_br}* de 2000 a 2023')
        st.plotly_chart(fig_hm2_br, use_container_width=True)



with tabs[3]:
    col_creditos1, col_creditos2 = st.columns([1, 1], gap='large')

    col_creditos1.subheader('Founded by [IRB(Re)](https://www.irbre.com/)')
    col_creditos1.caption('A leading figure in the Brazilian reinsurance market, with over 80 years of experience and a complete portfolio of solutions for the market.')
    col_creditos1.image('irb.jpg', use_column_width=True)

    col_creditos2.subheader('Developed by Instituto de Riscos Climáticos')
    col_creditos2.markdown('''
    **Supervisors:** Carlos Teixeira, Reinaldo Marques & Roberto Westenberger

    **Researchers:** Luiz Otávio & Karoline Branco

    **Data Scientists:**  Lucas Lima & Paulo Cesar
                        
    **Risk Scientists:** Ana Victoria & Beatriz Pimenta
                        
    #### Source
    - **The Emergency Events Database (EM-DAT)** , Centre for Research on the Epidemiology of Disasters (CRED) / Université catholique de Louvain (UCLouvain), Brussels, Belgium – [www.emdat.be](https://www.emdadt.be/).
    - **Atlas Digital de Desastres no Brasil** - [www.atlasdigital.mdr.gov.br/](http://atlasdigital.mdr.gov.br/).
    ''')
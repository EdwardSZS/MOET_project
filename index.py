import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pydeck as pdk
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, roc_auc_score


def main():

    page=st.sidebar.selectbox(
        "Selecciona una página",
        (
            "Inicio",
            "Descripción",
            "Resultados",
            "Conclusiones",
            "Proyectos futuros",
            "Referencias"

        )

    )
    page_=st.session_state
    if page == "Inicio":
        inicio()
    elif page == "Descripción":
        descripcion()
        series_de_tiempo()
    elif page == "Resultados":
        modelos()
    elif page ==  "Conclusiones":
        conclusiones()
       

def inicio():
    st.title("PROPUESTA DE UN MODELO PARA CLASIFICAR O AGRUPAR LA CONFIANZA DE INCENDIO EN UBICACIONES DE COLOMBIA MEDIANTE INFORMACIÓN DE PUNTOS DE CALOR")
    st.header("Objetivo general")
    st.subheader("Proponer un modelo que permita estimar la confianza de incendio en ubicaciones especificas de Colombia mediante información de puntos de calor, esto con el fin de brindar una herramienta que ayude a preparar al equipo de bomberos y entes ambientales ante posibles eventos provocados por el fenomeno del niño.")
    st.subheader("Objetivos específicos:")
    st.markdown(" **1.** Obtener información o el dataset para  entrenar el modelo.")
    st.markdown(" **2.** Identificar y analizar las variables relevantes para el modelo.")
    st.markdown(" **3.** Aplicar y evaluar modelos para clasificar o agrupar la confianza de incendio.")
    st.markdown(" **4.** Seleccionar el mejor modelo que caracterise la confianza del punto de calor para Colombia")
    map()
@st.cache_data
def map():
    df=load_data()
    # Define the links

    df['brightness']-=272.15
    df1=pd.DataFrame()
    df1['frp']=df['frp']
    df1['latitude']=df['latitude']
    df1['longitude']=df['longitude']
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=9.96935,
            longitude=-73.99149	,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
               data=df1,
               get_position='[longitude, latitude]',
               radius=150,
               elevation_scale=50,
               elevation_range=[30, 90],
               pickable=True,
               extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df1,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=150,
            ),
        ],
    ))

@st.cache_data
def load_data():
    links = [
    "https://drive.google.com/file/d/1aJ1FPuVaw-p4oL7jMQNcaZ5NSMz8RRdE/view?usp=drive_link",
    "https://drive.google.com/file/d/11WSLG7bLCFwDdbuCUBGoo3lXqwmUVyAC/view?usp=drive_link",
    "https://drive.google.com/file/d/1qn-QLO7HjpoXD5Xf9ZYsOxuJw3ypUaIB/view?usp=drive_link",
    "https://drive.google.com/file/d/17kG4qzSicRAZ5frEkcs-r9rbDPVgL_Q2/view?usp=drive_link"
    ]

    # Function to extract file ID from link
    

    # Read each CSV into a dataframe and store them in a list
    dfs = []
    for link in links:
        file_id = get_file_id(link)
        download_link = get_download_link(file_id)
        df = pd.read_csv(download_link, sep=",")
        dfs.append(df)

    # Concatenate all dataframes in the list
    df = pd.concat(dfs, ignore_index=True)
    return df


def get_file_id(link):
        return link.split('/')[-2]

    # Function to construct download link
def get_download_link(file_id):
        return f'https://drive.google.com/uc?id={file_id}'

def descripcion():
    st.title("Características de los satélites y del dataset")
    st.subheader("La información para el modelo fue obtenida directamente de la NASA por sus sensores satelitales VIIRS y MODIS que hacen capturas de puntos de calor al rededor del mundo.")
    st.image('Sensores.jpg', caption='Características de sensores de cada satélite.')
    st.markdown("Punto de calor: Cualquier anomalía térmica presente en la tierra. (incendio, volcanes, costa afuera u otra fuente terrestre estática)")
    st.markdown("Un punto de calor activo representa el centro de un píxel marcado que contiene 1 o más focos/incendios en llamas activas. En la mayoría de los casos, los puntos representan incendios, pero a veces pueden representra cualquier anomalia termica como una erupción volcánica  (NASA).")
    st.markdown("Consideramos que los datos actuales tienen una calidad suficientemente buena para utilizarlos en aplicaciones de gestión de incendios y estudios científicos (NASA).")					
    st.markdown("https://firms.modaps.eosdis.nasa.gov/map/#d:24hrs;@0.0,0.0,3.0z")				
					
					
def series_de_tiempo():
    st.title("Análisis de series de tiempo")
   
    df=load_data()
    # Define the links

    df['brightness']-=272.15
    
    df1=df
    df2=df
    def agregar_dos_puntos(valor):
        return valor[:-2] + ":" + valor[-2:]


    df['acq_time']=df['acq_time'].astype(str)
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_dos_puntos(x))

    def agregar_minutos_cero(valor):
        return valor + ":00"

    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_minutos_cero(x))
    df1['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'])
    
    df1 = df1.sort_values(by='datetime')
    df1=df1.groupby(by="datetime").agg({'brightness': 'mean'}).reset_index()
    band=0
    if st.sidebar.button('Horario') and band == 0:
        st.subheader("Serie de tiempo frecuencia horaria")
        band=plot_daily(df1,df2)
    if st.sidebar.button('Diario') and band == 0:
        st.subheader("Serie de tiempo frecuencia diaria")
        band=plot_month(df2,df1)
@st.cache_resource(experimental_allow_widgets=True)
def plot_daily(df1,df2):
    count=0
    fig = go.Figure(data=go.Scatter(x=df1['datetime'], y=df1.brightness, mode='lines+markers'))
    fig.update_layout(title='Time Series',
                      xaxis_title='Datetime',
                      yaxis_title='Index')
    st.plotly_chart(fig, use_container_width=True)
    st.header("Estimar las autocorrelacciones")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 8))

    plot_acf(df1['brightness'], lags=30, ax=axes[0])
    axes[0].set_title('Autocorrelation')

    plot_pacf(df1['brightness'], lags=30, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation')
    st.pyplot(fig1)
    return count
@st.cache_resource(experimental_allow_widgets=True)
def plot_month(df2,df1):
    count_1=0
    df2['datetime'] = pd.to_datetime(df2['acq_date'])
    df2 = df2.sort_values(by='datetime')
    df2=df2.groupby(by="datetime").agg({'brightness': 'mean'}).reset_index()
    fig = go.Figure(data=go.Scatter(x=df2['datetime'], y=df2.brightness, mode='lines+markers'))
    fig.update_layout(title='Time Series',
                  xaxis_title='Datetime',
                  yaxis_title='Brightness')
    st.plotly_chart(fig, use_container_width=True)
    fig1, axes = plt.subplots(1, 2, figsize=(10, 8))

    plot_acf(df2['brightness'], lags=15, ax=axes[0])
    axes[0].set_title('Autocorrelation')

    plot_pacf(df2['brightness'], lags=15, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation')
    st.pyplot(fig1)

    return count_1
def modelos():

    st.title("Análisis de modelos supervisados y no supervisados")
    supervisado()

def intervalo(valor):
    if int(valor) <=3 and int(valor) >=0:
      return "[0-3]"
    if int(valor) <=7 and int(valor) >=4:
      return "[4-7]"
    if int(valor) <=11 and int(valor) >=8:
      return "[8-11]"
    if int(valor) <=15 and int(valor) >=12:
      return "[12-15]"
    if int(valor) <=19 and int(valor) >=16:
      return "[16-19]"
    else:
      return "[20-23]"
def agregar_dos_puntos(valor):
  return valor[:-2] + ":" + valor[-2:]

def agregar_minutos_cero(valor):
  return valor + ":00"
def reemplazar(valor):
    if valor == "n":
      return valor
    if valor == "h":
      return valor
    if valor == "l":
      return valor
    elif int(valor) >=80:
      return "h"
    elif int(valor) <80 and int(valor) >=30:
      return "n"
    else:
      return "l"
    
def supervisado():
    df=load_data()


    # Aplicar la función a la columna 'tiempo'
    df['acq_time']=df['acq_time'].astype(str)
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_dos_puntos(x))
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_minutos_cero(x))
    #Se convierte la variable a °c
    df['brightness']-=272.15
    df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'])
    df['confidence'] = df['confidence'].apply(lambda x: reemplazar(x))
    df_regresionlogistica = df
    #df_regresionlogistica['acq_time'] = pd.to_datetime(df_regresionlogistica['acq_time'])
    #print(df_regresionlogistica.info())
    df_regresionlogistica['datetime'] -= pd.Timedelta(hours=5)
    
    df_regresionlogistica['hora'] = df_regresionlogistica['datetime'].dt.hour
   

    # Aplicar la función a la columna
    df_regresionlogistica['intervalo_hora'] = df_regresionlogistica['hora'].apply(lambda x: intervalo(x))
    
    df_regresionlogistica_1 = pd.get_dummies(df_regresionlogistica, columns=['intervalo_hora'])
    
    numeric_variables = df_regresionlogistica_1[["latitude", "longitude", "brightness", "frp","intervalo_hora_[0-3]","intervalo_hora_[12-15]","intervalo_hora_[20-23]","intervalo_hora_[8-11]"]]
    correlation_matrix = numeric_variables.corr()  # Matriz de correlación

    # Visualización de la matriz de correlación
    fig1, axes = plt.subplots( figsize=(6, 4))
   
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Matriz de Correlación")
    st.pyplot(fig1)
    df_regresionlogistica_2=df_regresionlogistica_1.drop(["acq_time","datetime","confidence","hora","intervalo_hora_[0-3]","brightness","scan","track","acq_date","satellite","instrument","version","bright_t31","daynight"], axis=1)
    x_regresionlogistica=df_regresionlogistica_1.drop(["acq_time","datetime","confidence","hora","intervalo_hora_[0-3]","brightness","scan","track","acq_date","satellite","instrument","version","bright_t31","daynight"], axis=1)
    y_regresionlogistica=df_regresionlogistica_1['confidence']
    print("hora",y_regresionlogistica)
    xl_train, xl_test, yl_train, yl_test = train_test_split(x_regresionlogistica, y_regresionlogistica, test_size=0.3, random_state=42)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    result=model.fit(xl_train, yl_train)
    class_weights = {"h": 10, "l": 6, "n": 4}  # Ejemplo de asignación de pesos

    # Entrena el modelo de regresión logística multinomial con reponderación de instancias
    # Entrena el modelo de regresión logística multinomial con reponderación de instancias
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=class_weights)
    model.fit(xl_train, yl_train)
    yl_pred = model.predict(xl_test)
    columnas=yl_test.unique()
    conf_matrix = confusion_matrix(yl_test, yl_pred)
    fig2, axes = plt.subplots( figsize=(4, 2))
    #plt.figure(figsize=(4,2))
    sns.heatmap(conf_matrix, annot=True, fmt='g',xticklabels=columnas, yticklabels=columnas)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    st.pyplot(fig2)
    report = classification_report(yl_test, yl_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.table(df_report)
def conclusiones():
    st.title("Conclusiones")


if __name__== "__main__":
    main()
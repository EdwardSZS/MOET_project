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
            "Proyectos",
            "Referencias"

        )

    )
    page_=st.session_state
  
       
    st.sidebar.subheader("💬 Chatbot")
    user_input = st.sidebar.text_input("Tú: ", "")

    # Generate response based on user input
    if user_input:
        response = generate_response(user_input)
        page=user_input
        st.sidebar.write("Chatbot:", response)


    if page == "Inicio":
        inicio()
    elif page == "Descripción":
        descripcion()
        info()
        series_de_tiempo()
    elif page == "Resultados":
        modelos()
    elif page ==  "Conclusiones":
        conclusiones()
    elif page == "Proyectos":
        proyectos() 
    elif page == "Referencias":
        referencias()
        
def inicio():
    st.title("PROPUESTA DE UN MODELO PARA DETERMINAR LA CONFIANZA DE INCENDIO EN UBICACIONES DE COLOMBIA")
    st.divider()
    st.header("Objetivo general")
    st.subheader("Proponer un modelo que permita estimar la confianza de incendio en ubicaciones de Colombia mediante información de puntos de calor, esto con el fin de brindar una herramienta que ayude a preparar al equipo de bomberos y entes ambientales ante posibles eventos provocados por el fenomeno del niño.")
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
    st.image('MODIS.jpg', caption='Características de la muestra escaneada por el satélite.')
    st.image('Viirs.jpg', caption='Características de la muestra escaneada por el satélite.')
    st.markdown("Punto de calor: Cualquier anomalía térmica presente en la tierra. (incendio, volcanes, costa afuera u otra fuente terrestre estática)")
    st.markdown("Un punto de calor activo representa el centro de un píxel marcado que contiene 1 o más focos/incendios en llamas activas. En la mayoría de los casos, los puntos representan incendios, pero a veces pueden representra cualquier anomalia termica como una erupción volcánica  (NASA).")
    st.markdown("Consideramos que los datos actuales tienen una calidad suficientemente buena para utilizarlos en aplicaciones de gestión de incendios y estudios científicos (NASA).")					
    st.image('Colombia.png', caption='El mapa de incendios identificados por el IDEAM.',use_column_width='auto')
    st.markdown("https://firms.modaps.eosdis.nasa.gov/map/#d:24hrs;@0.0,0.0,3.0z")				
					
def info():
    st.subheader("Descripción del tipo de datos")
    df=load_data()
    df['acq_time']=df['acq_time'].astype(str)
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_dos_puntos(x))
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_minutos_cero(x))
    #Se convierte la variable a °c
    df['brightness']-=272.15
    df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'])
    df['confidence'] = df['confidence'].apply(lambda x: reemplazar(x))
    st.markdown("Descripción del dataset")
    inf=df.info()
    print(inf)

    st.write(inf)
    st.write(df.describe())
   
    df['datetime'] -= pd.Timedelta(hours=5)
    fig, axes = plt.subplots(figsize=(10,6))
    fig = px.box(df, x="confidence", y="frp", points="all")
    st.plotly_chart(fig, use_container_width=True)

def referencias():
    st.header("Enlaces de consulta y preparación")
    st.markdown("1: https://firms.modaps.eosdis.nasa.gov/map/#d:2024-04-14..2024-04-15;@-73.03,4.17,14.00z")
    st.divider()
    st.markdown("2: https://www.earthdata.nasa.gov/faq/firms-faq#ed-fire-on-ground")  
    st.divider()
    st.markdown("3: Documents sensor viirs: https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms/vnp14imgtdlnrt")
    st.divider()
    st.markdown("4: Documentación sensor Modis: https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms/mcd14dl-nrt")

def generate_response(input_text):
    # Define navigation responses based on user input
    if "inicio" in input_text.lower():
        return "Estás en la página de Inicio. ¿En qué más puedo ayudarte?"
    elif "descripción" in input_text.lower():
        return "Estás en la página de Descripción. ¿Necesitas más información sobre algún aspecto específico?"
    elif "resultados" in input_text.lower():
        return "Estás en la página de Resultados. ¿Hay algo en particular que te gustaría revisar?"
    elif "conclusiones" in input_text.lower():
        return "Estás en la página de Conclusiones. ¿Te gustaría profundizar en algún punto en particular?"
    elif "proyectos" in input_text.lower():
        return "Estás en la página de Proyectos. ¿Qué proyecto te gustaría explorar?"
    elif "referencias" in input_text.lower():
        return "Estás en la página de Referencias. ¿Buscas alguna referencia específica?"
    else:
        return "Lo siento, no entendí. ¿Puedes ser más específico?"

def proyectos():
    st.header("Avances futuros")
    st.markdown("1. Estimar el área quemada a partir de datos confiables, ya que la NASA recomienda no hacer estas estimaciones con la información proveniente de los puntos de calor detectados por los sensores VIIRS y MODIS. La intención de este proyecto sería para complementar las herramientas que ayuden con la preparación ante posibles eventos de incendio.")
    st.divider()
    st.markdown("2. Fortalecer la herramienta interactiva y buscar crear una api para mayor integridad con otras apps, como las del IDEAM. ")

    
    
    
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
    df1=df1.groupby(by="datetime").agg({'brightness': 'mean','frp': 'mean'}).reset_index()
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
    st.markdown("Análisis para la temperatura")
    fig = go.Figure(data=go.Scatter(x=df1['datetime'], y=df1.brightness, mode='lines+markers'))
    fig.update_layout(title='Time Series',
                      xaxis_title='Datetime',
                      yaxis_title='brightness')
    st.plotly_chart(fig, use_container_width=True)
    st.header("Estimar las autocorrelacciones")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 8))

    plot_acf(df1['brightness'], lags=30, ax=axes[0])
    axes[0].set_title('Autocorrelation')

    plot_pacf(df1['brightness'], lags=30, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation')
    st.pyplot(fig1)
    st.markdown("Análisis para el FRP")
    fig2 = go.Figure(data=go.Scatter(x=df1['datetime'], y=df1.frp, mode='lines+markers'))
    fig2.update_layout(title='Time Series',
                      xaxis_title='Datetime',
                      yaxis_title='frp')
    st.plotly_chart(fig2, use_container_width=True)
    st.header("Estimar las autocorrelacciones")
    fig3, axes = plt.subplots(1, 2, figsize=(10, 8))

    plot_acf(df1['frp'], lags=30, ax=axes[0])
    axes[0].set_title('Autocorrelation')

    plot_pacf(df1['frp'], lags=30, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation')
    st.pyplot(fig3)
    return count
@st.cache_resource(experimental_allow_widgets=True)
def plot_month(df2,df1):
    count_1=0
    df2['datetime'] = pd.to_datetime(df2['acq_date'])
    df2 = df2.sort_values(by='datetime')
    df2=df2.groupby(by="datetime").agg({'brightness': 'mean','frp':'mean'}).reset_index()
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

    st.markdown("Análisis para el FRP")
    fig2 = go.Figure(data=go.Scatter(x=df2['datetime'], y=df2.frp, mode='lines+markers'))
    fig2.update_layout(title='Time Series',
                      xaxis_title='Datetime',
                      yaxis_title='frp')
    st.plotly_chart(fig2, use_container_width=True)
    st.header("Estimar las autocorrelacciones")
    fig3, axes = plt.subplots(1, 2, figsize=(10, 8))

    plot_acf(df2['frp'], lags=15, ax=axes[0])
    axes[0].set_title('Autocorrelation')

    plot_pacf(df2['frp'], lags=15, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation')
    st.pyplot(fig3)
    
    return count_1
def modelos():

    st.title("Análisis de modelos supervisados y no supervisados")
    st.subheader("Iteración de algunos modelos")
    st.image('modelos.jpg', caption='Muestras de algunos modelos')
    with st.expander("Modelo supervisado"):
        supervisado()
    with st.expander("Modelo No supervisado"):    
        nosupervisado()

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
#@st.cache_resource(experimental_allow_widgets=True)
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
    option=st.sidebar.multiselect(
        "Items probados en el modelo",
        (
            "latitude",
            "longitude",
            "brightness",
            "frp",
            "intervalo_hora_[0-3]",
            "intervalo_hora_[12-15]",
            "intervalo_hora_[20-23]",
            "intervalo_hora_[8-11]"

        ),("latitude",
            "longitude",
            "brightness",
            "frp",
            "intervalo_hora_[0-3]",
            "intervalo_hora_[12-15]",
            "intervalo_hora_[20-23]",
            "intervalo_hora_[8-11]")


    )

    df_regresionlogistica_0=df_regresionlogistica_1[option]
   

    numeric_variables = df_regresionlogistica_0
    correlation_matrix = numeric_variables.corr()  # Matriz de correlación

    # Visualización de la matriz de correlación
    fig1, axes = plt.subplots( figsize=(6, 4))
   
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Matriz de Correlación")
    st.subheader("Matriz de correlación general")
    st.pyplot(fig1)
    #df_regresionlogistica_2=df_regresionlogistica_0.drop(["acq_time","datetime","confidence","hora","intervalo_hora_[0-3]","brightness","scan","track","acq_date","satellite","instrument","version","bright_t31","daynight"], axis=1)
    x_regresionlogistica=df_regresionlogistica_0
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
    st.subheader("Matriz de confusión para la regresión logística ")
    conf_matrix = confusion_matrix(yl_test, yl_pred)
    fig2, axes = plt.subplots( figsize=(4, 2))
    #plt.figure(figsize=(4,2))
    sns.heatmap(conf_matrix, annot=True, fmt='g',xticklabels=columnas, yticklabels=columnas)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    st.pyplot(fig2)
    report = classification_report(yl_test, yl_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.subheader("Tabla del accuracy del modelo ")
    st.table(df_report)
@st.cache_resource 
def nosupervisado():
    df=load_data()


    # Aplicar la función a la columna 'tiempo'
    df['acq_time']=df['acq_time'].astype(str)
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_dos_puntos(x))
    df['acq_time'] = df['acq_time'].apply(lambda x: agregar_minutos_cero(x))
    #Se convierte la variable a °c
    df['brightness']-=272.15
    df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'])
    df['confidence'] = df['confidence'].apply(lambda x: reemplazar(x))
    dfU = df
   
    dfU['datetime'] -= pd.Timedelta(hours=5)

    confidence_mapping = {'l': 0, 'n': 1, 'h': 2}
    dfU['confidence_numeric'] = dfU['confidence'].map(confidence_mapping)
    X = dfU[['latitude', 'longitude', 'brightness', 'frp', 'confidence_numeric']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig3, axes = plt.subplots(figsize=(10,6))
    plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    st.pyplot(fig3)

    X_scaled = X_scaled.T
    #silhouette_score estima el número de cluster / son tres niveles de confianza
    # Set the number of clusters
    n_clusters = 3
    #1.1 Fuzziness parameter (m)
    # Apply Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_scaled, n_clusters, 1.1, error=0.005, maxiter=1000, init=None)

    # Assign clusters
    cluster_membership = np.argmax(u, axis=0)

    # Plotting the clusters
    fig4, axes = plt.subplots(figsize=(10,6))
    plt.scatter(dfU['longitude'], dfU['latitude'], c=cluster_membership, cmap='viridis', s=50, alpha=0.5)
    plt.title('Fuzzy C-Means Clustering of Fire Incidents (Confidence Levels)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    st.pyplot(fig4)
    st.markdown("Descripción del score: ")
    st.write(fpc)
def conclusiones():
    st.title("Conclusiones")
    st.subheader("1: De los modelos de clasificacion utilizados, se considera que el mejor resultado lo obtuvo la regresion logistica sin reponderacion de instancias, pues, a pesar que el modelo no tiene capacidad para clasificar la confianza de incendio en la categoria baja, si obtiene las mejores precisiones en las categorias de alta y media confianza de incendio las cuales son las de interés.")
    st.divider()
    st.subheader("2: En conclusión, al momento de generar un modelo no supervisado como el fuzzy c means, permite agrupar los datos con similitudes en características, lo cual ha permitiido analizar la intensidad en la radiación segregada por regiones como la orinoquía y al región andina. ")

if __name__== "__main__":
    main()

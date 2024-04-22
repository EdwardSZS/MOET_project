import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf




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
        series_de_tiempo()
    elif page == "Resultados":
        modelos()
    elif page ==  "Conclusiones":
        conclusiones()
       

def inicio():
    st.title("Las perspectivas del fénomeno del niño")
    st.header("Objetivos")
    st.subheader("1.")
    st.subheader("2.")

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
    if st.sidebar.button('Diario') and band == 0:
        st.subheader("Serie de tiempo frecuencia diaria")
        band=plot_daily(df1,df2)
    if st.sidebar.button('Mensual') and band == 0:
        st.subheader("Serie de tiempo frecuencia mensual")
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

    st.title("Análisis de modelos suspervisados y no supervisados")

def conclusiones():
    st.title("Conclusiones")


if __name__== "__main__":
    main()
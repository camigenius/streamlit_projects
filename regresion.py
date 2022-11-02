from cgitb import small
from email.base64mime import header_length
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Regresion Lineal Simple",
    page_icon="wolf",
    layout='wide'
)

with st.container():

    
    col1, col2 = st.columns([3,1],gap='small')

    
    with col1:
        st.header('Trabajo Regresión Lineal Simple')
        st.subheader('Presentado por : Julio Mario cod y Camilo Franco cod')
#st.subheader('Julio Mario')
#st.subheader('Camilo Franco Guzman')
         
    with col2:
        st.image("logo_u.jpg",caption='Universidad Los Libertadores') 


st.write('---')

df= pd.read_csv('inmuebles_bogota_res.csv')
st.code("df= pd.read_csv('inmuebles_bogota_res.csv')")
st.write(df.head())
st.write(df.describe())

df['log_venta']=np.log10(df.mvalorventa)
df['log_marea']=np.log10(df.marea)
#st.write(df[['log_marea','log_venta']].head())

st.write('---')
with st.container():
    
    col1,col2,col3=st.columns([2,2,2],gap='small')


    with col1:        


        f,ax =plt.subplots(figsize=(7,5))
        sns.despine(f)
        sns.set_theme(style="ticks")
        sns.histplot(
            df,
            x='log_marea',hue='mtipoinmueble',
            multiple='stack',
            palette='light:m_r',
            edgecolor=".3",
            linewidth=.5,
            )

        st.pyplot(f)

    with col2:
        f,ax =plt.subplots(figsize=(7,5))
        sns.despine(f)
        sns.set_theme(style="ticks")
        sns.histplot(
            df,
            x='log_venta',hue='mtipoinmueble',
            multiple='stack',
            palette='pastel',
            edgecolor=".3",
            linewidth=.5,
            )

        st.pyplot(f)      

    with col3:
        source = df
       
        fig=alt.Chart(df).mark_point().encode(
        x='log_marea:Q',
        y='log_venta:Q',
        color='mtipoinmueble:N')

        st.altair_chart(fig)







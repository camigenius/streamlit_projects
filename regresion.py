from cgitb import small
from email.base64mime import header_length
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import streamlit as st
import altair as alt
from streamlit_lottie import st_lottie
import requests
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import shapiro
import pingouin as pg
from PIL import Image # imagenes

st.set_page_config(
    page_title="Regresion Linear Simple",
    page_icon="wolf",
    layout='wide'
)


# Using "with" notation

st.subheader('App Developed by:')

with st.container():

    
    col1, col2 = st.columns([2,2],gap='small')

    
    with col1:
        
        st.markdown('### 🚀Julio Mario Duran Ramirez.')
        st.write('*Especialista en Estadística e Ingeniero Industrial* (🇨🇴).')
        
#st.subheader('Julio Mario')
#st.subheader('Camilo Franco Guzman')
         
    with col2:
        #st.image("logo_u.jpg",caption='Universidad Los Libertadores')
        st.markdown('### 🤖 Camilo Franco Guzmán.')
        st.markdown('*Especialista en Estadistica ,Economista y Científico de Datos* (🇨🇴).')

st.write('---')


with st.container():

    col1,col2=st.columns([1,3],gap="small")

    
    with col1:
        st.header('Simple linear Regression Model')
        st.markdown('*"Si torturas suficientemente a los datos, éstos terminarán por confesar :Gregg Easterbrook"*')
        
    
    
    with col2:
        
        st.latex(r'''y=\beta_{0} +\beta_{i}X_{i}+\epsilon _{i}''')
    
        def load_lottieurl(url):
            r=requests.get(url)
            if r.status_code !=200:
                return None
            return r.json()
        Lottie_coding=load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_9azkhcpb.json")
        st_lottie(Lottie_coding,height=280,key='coding')



   
st.write('---')



tab1, tab2, tab3 = st.tabs([" ⬇️ Como usar es App               "," 👇 click Aqui !¿Quieres saber un poco del Modelo?", "Click Aqui💻Como fúe construida esta APP"])

with tab2:
   st.markdown("La Regresión Lineal o linear regression es una técnica paramétrica utilizada para predecir variables continuas dependientes, dado un conjunto de variables independientes (para el modelo simple solo una) a través del metodo OLS por sus siglas en Ingles 'Ordinary Least Square'.")
   st.markdown("Se puede usar para los casos donde quieras predecir alguna cantidad continua por ejemplo:")
   st.markdown("* 💵Predecir los ingresos de una empresa utilizando el gasto en publicidad como variable de predicion.")
   st.markdown("* 💊La presíon Arterial de acuerdo a una dosificación  un determinado medicamentoa a un paciente.")   
   st.markdown("* Cuando son Multiples variables predictoras  se llama Regresión  Multiple modelo que desarrollaré en otra app donde habrá ¡FUEGO!🧨🔥,oséa Machine Learning e Inteligencia Artificial")
   st.markdown("🏡Para nuestro caso queremos predecir el valor de un Inmueble, casa o apartamento en función del área en mts cuadrados basados en dataset de MetroCuadrado.com para la ciudad de Bogotá")
   st.latex(r'''y=\beta_{0} +\beta_{i}X_{i}+\epsilon _{i}''')
   

with tab1:
   
   with st.container():

    

    col1,col2=st.columns([1,3])

    with col1:
        
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        
    with col2:
        st.subheader("Usar esta App es muy fácil y Firulais lo sabe 🦴 ")
        st.markdown("##### Ve desplegando las diferentes opciones de la parte inferior⬇️")
        st.write("")
        st.markdown("##### ⬅️Por último a la izquierda de la página encontraras una *slider* o barrita que puedes deslizar.")
        st.write("##### Al deslizarla aumentara o disminuira el área en mts cuadrados y si señores como por arte de magia habrás pronosticado el valor del inmueble.")
        st.markdown("*Dato Curioso :🐶Según las malas lenguas el origen de la palabra Firulais deriva de la expresión en inglés free of lice que en español significa “libre de pulgas”.*")



with tab3:

   #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
   
   st.write("Esta app fue desarrollada en su mayor parte en lenguaje pyhton \
    by clicking on the following link: https://github.com/camigenius")
   st.write("Se utlizo paqueteria de modelado estadístico , tambien  de manipulacion y visualizacion de datos")

   st.code('''
        
        from cgitb import small
        from email.base64mime import header_length
        import pandas as pd
        import seaborn as sns
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        import streamlit as st
        import altair as alt
        from streamlit_lottie import st_lottie
        import requests
        from scipy import stats
        from scipy.stats import pearsonr
        from scipy.stats import shapiro
        import pingouin as pg
        from PIL import Image ''') 

st.write('---')
#with st.container():
    #col1 ,col2 = st.columns([4,2],gap='small')   
    
    #with col1 :

        #st.markdown('')
see_data=st.expander('Puedes dar click  aqui si deseas ver el Dataset de metrocuadrado.com !👉 ')
        
with see_data:
    df= pd.read_csv('inmuebles_bogota_res.csv')
    st.code("df= pd.read_csv('inmuebles_bogota_res.csv')")
    df=df.drop(['Unnamed: 0'], axis=1)
    st.write(df.head(5))

see_describe=st.expander('Que tal si observas algunas medidas Descriptivas 📊')        
with see_describe:
    st.code('df.describe()')
    st.write(df.describe().T)
    
with st.expander("👀 ¡Mira! Se ejecutó una tranformación logaritmica  necesaria dada la escala de valores  entre área y valor venta"):
    col1,col2,col3= st.columns([0.7,0.1,0.5],gap='large')

    with st.container():
        with col1:
            st.write('')
            st.markdown('* Hasta el momento no se puede ver una relación grafica evidente entre área y valor venta')
            st.write('')
            source = df
            fig=alt.Chart(df).mark_point().encode(
            x='marea:Q',
            y='mvalorventa:Q',
            color='mtipoinmueble:N')
            st.altair_chart(fig,use_container_width=True)
            


        with col2:
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('⏩⏩')


        with col3:
            st.write('Agregamos dos Features nuevas con la transformacion Logarítmica')
            df['log_venta']=np.log10(df.mvalorventa)
            df['log_marea']=np.log10(df.marea)
            st.write(df[['log_marea','log_venta','mvalorventa','marea']].head(3))
            st.code("df['log_venta']=np.log10(df.mvalorventa)")
            st.code("df['log_marea']=np.log10(df.marea)")
            st.write('')
            

        
    # with col3:    
    #     st.write('* Una vez realizada la Transformación log se evidenccia la relación')
    #     source = df             
    #     fig=alt.Chart(df).mark_point().encode(
    #     x='log_marea:Q',
    #     y='log_venta:Q',
    #     color='mtipoinmueble:N')

    #     st.altair_chart(fig)
agree = st.checkbox('¿⬅️ Click  si  eres curioso 🕵️y quieres ver el gráfico después de la tranformación Logarítmica 🤓?')


if agree:
          
      
    st.write('#### Voilà! Ahora si se observa una relación evidente 😎 (Te recomiendo expandir el Gráfico se ve expectacular 👉)')        
    source = df
    fig=alt.Chart(df).mark_point().encode(
    x='log_marea:Q',
    y='log_venta:Q',
    color='mtipoinmueble:N')
    st.altair_chart(fig,use_container_width=True)

    #st.write('Correlación Pearson: ', df['log_marea'].corr(df['log_venta'], method='pearson'))
    st.write('Correlación spearman: ', round(df['log_marea'].corr(df['log_venta'], method='spearman'),2))
    st.write('Correlación kendall: ', round(df['log_marea'].corr(df['log_venta'], method='kendall'),2))
    

with st.expander('🧑‍🎄🎄Debe ser navidad por que aqui aparecen un par de campanas🔔'):
    st.write('')
    st.markdown('Bueno no son de navidad lo acepto!!, pero son distribuciones con forma de campana (normales aprentemente🤫) de Nuestro amigo Gauss Jordan el Principe de las matemáticas!!')
    st.write('')
    
    with st.container():
        col1,col2=st.columns([2,2],gap='small')
        
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
    st.markdown('Y cuando todo "parecia" "Normal"😂! ; pero noooo! ,según don Shapiro')
    #shapiro_test1 = stats.shapiro(df.log_marea)
    #st.write(f"Variable Área: {shapiro_test1}")
    estadistico,p_value=shapiro(df.log_venta)
    #shapiro_test = stats.shapiro(df.log_venta)
    st.write('#### Estadístico Shapiro =%.3f,p_value=%.3f'% (estadistico,p_value))
    st.code('estadistico,p_value=shapiro(df.log_venta)')
    st.code("Estadistico = %.3f,p_value = %.3f'% (estadistico,p_value)")                 


#X = sm.add_constant(df['log_marea'])
#y=df['log_venta']

#model = sm.OLS(y, X)
#results = model.fit()
#print(results.summary())

df2=df[(df['log_marea']>1)&(df["log_venta"]<11)&(df["log_venta"]>7)]
model=smf.ols(formula='log_venta ~ log_marea',data=df2).fit()
st.write('---')

with st.sidebar:
    st.markdown('### Pronóstica el valor de un  Inmueble 🏡')
    area=st.slider('Selecciona el valor del Area (mts cuadrados)',min_value=0.0,max_value=5000.0,step=10.0)
    st.write('Elegiste una area de : ',area,"mts cuadrados")
    
    st.write('---')
    coef= model.params
    coef_bo=coef[0]
    coef_b1=coef[1]
    y_estimado=(10**coef_bo)*(area**coef_b1)
    st.write('Intercepto bo =',round(coef_bo,2))
    st.write('Coeficiente b1 = ',round(coef_b1,2))
    st.write('EL Valor del inmuebles es: $ ')
    st.write(round(y_estimado/1000000,0),'Millones de Pesos')



if st.button('!NUNCA PERO NUNCA LE DES CLICK A ESTE BOTON 🚨🚫!😱'):
    #st.subheader('Bueno eres demasiado curioso lo sabia!!')
    st.title('¡BOOOOOM!💣')
    st.image("boom.jpg")
    #st.subheader('Te lo dije ahora es resposabilidad tuya entender todo esto')
    
    st.markdown('* 🤫 Bueno como no obedeciste hubo que suprimier unos cuantos outliers para que todo funcionara ✂️ 🤭 *' )

    
    st.code("df2=df[(df['log_marea']>1)&(df['log_venta']<11)&(df['log_venta']>7)]")

    st.code("smf.ols(formula='log_venta ~ log_marea',data=df2).fit()")

    st.subheader("😭Y que es todo esto!")
    st.write("Tranquilo para el entendimiento practico de la APP nos podemos quedear por ahora con el R-square que aparece en la tabla de abajo⬇️.")
    st.write("El R-square  es una medida estadística de qué tan cerca están los datos de la línea de regresión ajustada. También se conoce como coeficiente de determinación, o coeficiente de determinación múltiple si se trata de dos o mas variables")
    st.write("Los valores que puede tomar el R-square van de 0 hasta 1 entre mas cercano a 1 mejor para nuestro modelo.")
    st.markdown("*Pero ojo 👁️ el R-square tiene sus limitaciones y no es lo único que debemos analizar en un modelo.*")
    st.write(model.summary())

    #st.expander('Bueno que más da ahora vas a ver el modelo')


    st.write('---')
    st.write("##### !En este gráfico vemos la representación de la Recta de regresión que mejor se ajsuta a los datos!")
    st.write('##### Esto a tráves de una función de  minimizacion que no es otra mas que una derivada (OLS)')

    sns.set_theme(style="whitegrid")
    log_area = np.log10(area)
    g,fig=plt.subplots(figsize=(15,7))
    ax=plt.scatter(x=df2['log_marea'].values,y=df2['log_venta'].values,color='g',marker=".")
    ax=plt.plot(df2['log_marea'].values,model.fittedvalues,color='m')
    ax=plt.scatter(x=log_area,y=coef_bo+coef_b1*log_area,color='r',marker='D')
    ax=plt.xlabel('log_marea')
    ax=plt.ylabel('log_venta')
    ax=plt.title('Recta de regresión (Mínimos Cuadrados Ordinarios OLS)')
    st.pyplot(g)

    #sns.set_theme(style="whitegrid")
    #g,fig=plt.subplots(figsize=(15,7))
    #ax=plt.scatter(x=df2['log_marea'].values,y=df2['log_venta'].values,color='g',marker="*")
    #ax=plt.plot(df2['log_marea'].values,model.fittedvalues,color='r')
    #ax=plt.xlabel('log_marea')
    #ax=plt.ylabel('log_venta')
    #ax=plt.title('Recta de regresión (Mínimos Cuadrados Ordinarios)')
    #st.pyplot(g)






#coef_bo+coef_b1*2.69
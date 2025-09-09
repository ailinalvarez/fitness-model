####

# ETAPA 1

####

#importamos librerias
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#cargamos datos 
data = pd.read_csv('gym_churn_us.csv')

####

# ETAPA 2

####


#observo tamaño del data set y una muestra aleatoria de 5 filas
print("Dimensiones del dataset:", data.shape)
print(data.sample(5))

#cambio nombre de columnas para que al graficar sea más facil ver sus nombres:
data.columns = ['gender', 'near', 'partner', 'promo', 'phone', 'contract_for', 'group', 'age', 'extra_cost', 'ending_in', 'lifetime', 'avg_freq', 'avg_month', 'churn']

#información resumida del dataset y observo si hay valores nulos
data.info()

# eliminamos las filas con los valores duplicados e imprimimos el tamaño del DataFrame actualizado
data.drop_duplicates(inplace = True)
print(data.shape)

#cambio el tipo de datos de la columna 'ending_in' a entero
data['ending_in'] = data['ending_in'].astype('int')

#observo valores promedio y la desviación estándar con el método describe()
print(data.describe())

#Observo los valores medios de las características en dos grupos: para las personas que se fueron  y para las que se quedaron en el gimnasio
churn = data.groupby('churn').mean()
display(churn)

#Graficamos histogramas de barras y distribuciones de características para aquellas personas que se fueron y para las que se quedaron.

#primero creo 2 dataframes, uno para los que se quedaron y otro para los que se fueron
data_cancelled = data[data['churn'] == 1]
data_continue = data[data['churn'] == 0]

#defino una función para graficar histogramas y distribuciones

def histograma_comparativo(columna, title, x,y):
    plt.figure(figsize=(15, 5))
    sns.histplot(data=data_continue[columna], label='Continúan', fill=True, color='purple', kde=True, bins=x)
    sns.histplot(data=data_cancelled[columna], label='Cancelaron', fill=True, color='red', kde=True, bins=y)
    plt.title(f'Distribución comparativa: {title}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
histograma_comparativo('gender', 'género de los usuarios', 10, 10)
# 0 corresponde a hombres y 1 corresponde a mujeres

histograma_comparativo('near', 'clientes que residen cerca del gimnasio', 10, 10)
# 0 corresponde a que no residen cerca del gimnasio y 1 que sí residen cerca del gimnasio

histograma_comparativo('partner', 'clientes que trabajan en una compañía asociada', 10, 10)
# 0 corresponde a que no trabajan en una compañía asociada y 1 que sí trabajan en una compañía asociada

histograma_comparativo('promo', 'usuarios que inscribieron con un código promocional de un amigo', 10, 10)
# 0 corresponde a que no inscribieron con un código promocional de un amigo y 1 que sí inscribieron con un código promocional de un amigo

histograma_comparativo('phone', 'usuarios que aportaron su número de teléfono', 10, 10)
# 0 corresponde a que no aportaron su número de teléfono y 1 que sí aportaron su número de teléfono

histograma_comparativo('age', 'edad de los usuarios', 41, 41)

histograma_comparativo('lifetime', 'tiempo (en meses) que el usuario ha sido cliente del gimnasio', 30, 9)

histograma_comparativo('contract_for', 'período del contratos', 24, 24)

histograma_comparativo('ending_in', 'meses que faltan para finalizar el contrato', 12, 12)

histograma_comparativo('group', 'participación en actividades grupales', 10, 10)
# 0 corresponde a que no participaron en actividades grupales y 1 que sí participaron en actividades grupales

histograma_comparativo('avg_freq', 'frecuencia promedio de visitas al gimnasio en su subscripción total', 12, 14)

histograma_comparativo('avg_month', 'frecuencia promedio de visitas por semana en el mes actual', 12, 14)

histograma_comparativo('extra_cost', 'total de dinero gastado en otros servicios del gimnasio', 50, 50)

#Construimos y representamos una matriz de correlación para todo el conjunto de datos

cm = data.corr()
fig, ax = plt.subplots(figsize=(15, 10))

# Generamos un mapa de calor
sns.heatmap(cm, annot=True, fmt=".2f", cmap='rocket_r', square=True, ax=ax, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación')
plt.xticks(rotation=45)
plt.show()


#####

# ETAPA 3

#####

#Creamos un modelo de clasificación binaria para clientes donde la característica objetivo es la marcha del usuario o la usuaria el mes siguiente.

#Primero separamos la variable objetivo (y) del resto de las características (X)
y = data['churn']
X = data.drop('churn', axis=1)

#Dividimos los datos en conjuntos de entrenamiento y validación

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Entrenamos el modelo en el set de entrenamiento con regresión logística

#definimos algoritmo del modelo
model_lr = linear_model.LogisticRegression(max_iter=1000)

#entramos el modelo
model_lr.fit(X_train, y_train) 

#utilizamos el modelo para hacer predicciones en el conjunto de prueba
predictions_lr = model_lr.predict(X_test)
probabilities_lr = model_lr.predict_proba(X_test)[:,1]

#Evaluamos la exactitud, precisión y recall para el modelo:

print('Exactitud: {:.2f}'.format(accuracy_score(y_test, predictions_lr)))
print('Precisión: {:.2f}'.format(precision_score(y_test, predictions_lr)))
print('Recall: {:.2f}'.format(recall_score(y_test, predictions_lr)))



#Entrenamos el modelo en el set de entrenamiento con bosque aleatorio.

# definimos el algoritmo para el modelo
model_rf = RandomForestClassifier(random_state=0)

# entrenamos el modelo utilizando los datos de entrenamiento
model_rf.fit(X_train, y_train)

# hacemos pronósticos con el modelo entrenado
predictions_rf = model_rf.predict(X_test)

#Evaluamos la exactitud, precisión y recall para el modelo:

print('Exactitud: {:.2f}'.format(accuracy_score(y_test, predictions_rf)))
print('Precisión: {:.2f}'.format(precision_score(y_test, predictions_rf)))
print('Recall: {:.2f}'.format(recall_score(y_test, predictions_rf)))


###### 

# ETAPA 4

######


#estandarizamos los datos

# creamos un objeto de clase scaler (normalizador)
scaler = StandardScaler()

# entrenamos el normalizador y transformamos el conjunto de datos
X_sc = scaler.fit_transform(data.drop(columns = ['churn'])) 

#creamos una variable donde almacenamos la tabla con los haces de objetos vinculados
linked = linkage(X_sc, method = 'ward')

# graficamos el dendrograma
plt.figure(figsize=(15, 10))  
dendrogram(linked, orientation='top')
plt.title('Agrupación jerárquica para GYM')
plt.show()


#K-means agrupa objetos paso por paso, teniendo en cuenta los colores del dendrograma el número óptimo de clústeres sugerido es 4.

km = KMeans(n_clusters = 4, random_state=0) 

# aplicamos el algoritmo a los datos y formar un vector de clúster
labels = km.fit_predict(X_sc) 

#almacenamos las etiquetas de clúster en el campo de nuestro conjunto de datos
data['cluster_km'] = labels

# obtiene las estadísticas de los valores medios de las características por clúster
data_cluster = data.groupby(['cluster_km']).mean()

print(data_cluster.sort_values(by='churn', ascending=False))


#El valor de la silueta muestra hasta qué punto un objeto de un clúster es similar a su clúster, en vez de otro. El valor oscila entre -1 y 1, 
#Un valor cercano a 0 indica que el objeto está en el límite entre dos clústeres.

# Calculamos el valor de la silueta para el agrupamiento

print('Silhouette_score: {:.2f}'.format(silhouette_score(X_sc, labels)))

#Creamos función para representar gráficos de características pareadas para los clústeres
def show_clusters(df, x_name, y_name, cluster_name, w, l, titulo):
    plt.figure(figsize=(w, l))
    sns.scatterplot(x=df[x_name], y=df[y_name], hue=df[cluster_name], palette='rocket_r')
    plt.title('{}'.format(titulo))
    plt.show()
    
    
# Representamos el gráfico para las características pareadas de cluster

show_clusters(data, 'churn', 'near','cluster_km', 10, 5, 'Relacion de clústers respecto a la cercanía y su membresía (0 continuidad - 1 cancelación)')

show_clusters(data, 'churn', 'partner','cluster_km', 10, 5, 'Relacion de clústers respecto al trabajo y su membresía (0 continuidad - 1 cancelación)')

show_clusters(data, 'churn', 'promo','cluster_km', 10, 5, 'Relacion de clústers respecto a promoción y su membresía (0 continuidad - 1 cancelación)')

show_clusters(data, 'churn', 'group','cluster_km', 10, 5, 'Relacion de clústers respecto a clases grupales y su membresía (0 continuidad - 1 cancelación)')


show_clusters(data, 'churn', 'age','cluster_km', 15, 5, 'Relacion de clústers respecto a la edad y su membresía (0 continuidad - 1 cancelación)')

show_clusters(data, 'churn', 'extra_cost','cluster_km', 15, 5, 'Relacion de clústers respecto a costos extras y su membresía (0 continuidad - 1 cancelación)')

show_clusters(data, 'churn', 'avg_month','cluster_km', 15, 7, 'Relacion de clústers respecto a la asistencia semanal del corriente mes y su membresía (0 continuidad - 1 cancelación)') 

show_clusters(data, 'churn', 'lifetime','cluster_km', 15, 7, 'Relacion de clústers respecto a la antigüedad y su membresía (0 continuidad - 1 cancelación)') 



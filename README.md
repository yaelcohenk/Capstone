# Grupo Capstone - Gestión de Inventario

## Cosas por hacer e ideas

## Flujo lógico de las cosas

- Antes de analizar los datos y todo eso, tenemos que quitar todos los outliers. Posterior a haber
quitado los outliers, podemos empezar a analizar. En ese caso, tenemos que correr todo el resto de nuevo.
De los datos que nos interesan. En verdad esto de los outliers nos debería interesar solo para el forecasting.
En verdad el resto de cosas no depende tanto de esto.



## Papers

[Deep Reinforcement Learning for inventory control: A roadmap](https://www.sciencedirect.com/science/article/pii/S0377221721006111)

[Cooperative Multi-Agent Reinforcement Learning for Inventory Management](https://arxiv.org/abs/2304.08769)

[Deep Reinforcement Learning for inventory optimization with non-stationary uncertain demand](https://www.sciencedirect.com/science/article/abs/pii/S0377221723007646)

[Leadtime effects and policy improvement for stochastic inventory control with remanufacturing](https://www.sciencedirect.com/science/article/abs/pii/S0925527300001353)

[Using the Deterministic EOQ Formula in Stochastic Inventory Control](https://www.jstor.org/stable/2634597)

[Reinforcement Learning Approach for Multi-period Inventory with Stochastic Demand](https://link.springer.com/chapter/10.1007/978-3-031-08333-4_23)

[Continuous inventory control with stochastic and non-stationary Markovian demand](https://www.sciencedirect.com/science/article/abs/pii/S0377221718302327)

[Inventory Management with Stochastic Lead Times](https://pubsonline.informs.org/doi/abs/10.1287/moor.2014.0671)
[]()
[]()
[]()

### Dicho por profesor en la reunión 2



### Análisis de datos

- Hacer la filtración de datos. Ver si quizás hay datos con magnitudes muy distintas. Revisar los outliers y que hacer con ellos.
- Ver si los datos tienen alguna pillería, como valores negativos o cosas así.
- Ver como segmentar a los clientes, agruparlos. Ver también si hay ciertos productos que se compran en específico.
- Ver si la demanda por producto es intermitente o no. Hay distintos modelos de pronóstico.
- Ver si hay estacionareidad, si es que para cierta época ciertos productos se venden más o menos, cosas así.
Por ejemplo, ver semanal/mensualmente la demanda de los medicamentos, ser capaz de que por ejemplo, quizás en junio
se venden más respecto a todo el resto del año y de ahí desagregar por cada producto de medicamentos.
- Podría hacer el análisis del porcentaje de ganancias de cada uno de los 49 subgrupos de manera mensual y semanal.
 Es decir, para cada una de las semanas y para cada uno de los meses. Quizás podría hacer lo mismo con los 3 grupos grandes, no sé si vale la pena hacerlo con cada uno de los productos
- Ver si hay estacionariedad en las demandas porque en dicho caso, tendríamos que realizar un análisis por estación. Ver si tener modelos por tramos
- Graficar cosas desagregadas según distintos periodos de tiempo. Por tendencia mensual, semanal, diaria
- Fijarse en la tendencia, la idea es hacer el modelo de pronóstico de eso x semana, mientras más preciso mejor. Hacerlo por día nomás. Por precisión, idealmente sería mejor hacerlo producto a producto.
- Ver la importancia de lo que aporta en ganancia cada producto. Ver si por período de tiempo, o del total
- Podría hacer una matriz de correlaciones de forma año-mes, para ver que productos se venden en conjunto unos con otros




### Modelos de pronóstico

- Para los pronósticos se podría eventualmente intentar programar una RNN (LSTM).
- Autoregression (AR)
- Moving Average (MA)
- Autoregressive Moving Average (ARMA)
- Autoregressive Integrated Moving Average (ARIMA)
- Seasonal Autoregressive Integrated Moving Average (SARIMA)
- Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX)
- Vector Autoregression (VAR)
- Vector Autoregression Moving Average (VARMA)
- Vector Autoregression Moving Average with Exogenous Regressors (VARMAX)
- Simple Exponential Smoothing (SES)
- Holt Winter's Exponential Smoothing (HWES).

### Validez estadística de los datos

- No olvidar el test estadístico. Ajustar los datos a una distribución y tener en cuenta su valor 
Chi-Cuadrado y el test de Kolmogorov-Smirnov. Luego se podría hacer lo mismo con cada uno de los 49
subgrupos, para que esto sea más preciso. Hacerlo eventualmente con cada uno de los productos




### Resolución del problema


### Tips

- Una idea puede ser quizás agrupar los productos no tanto por temas de cantidad demandada, sino por el tipo
de demanda al que se enfrentan. Esto para posteriormente elegir correctamente el modelo de pronóstico.
- Categorizar la demanda, porque hay modelos que funcionan mejor que otros dependiendo del tipo de demanda.
- Ver los modelos ABC y Pareto
- Partir de lo más agregado a lo menos desagregado.
- Revisar cada uno de los productos, revisar cuantas veces se ha vendido cada uno de ellos,
si hay algunos que se dejaron de vender, cuando fue la última venta, ver si sacarlo si tiene pocos
datos.
- Se podría hacer un leadtime como variable aleatoria.
- Para validar modelo de pronóstico, 75% datos para entrenar y 25% validar
- Como presentar la información después, es importante eso.
- Centrarse en como armar los modelos de pronóstico
- Los KPI para la primera entrega van si o si, son las métricas de desempeño de la solución.
- No es mala idea resolver el problema con OD, al menos en un sentido aproximado. Ver si podemos utilizar RL
- Los KPI se asocian a la política de inventario, como una política es mejor que otra, bajo que criterios.
- Una vez categorizados los productos lo que se puede hacer es buscar modelos de pronóstico por cada uno y ahí iterar para el que minimiza el error.
- Se podría minimizar una ponderación entre el MASE, el MES Y EL R2, para elegir.
- Estudiar como funcionan los modelos de pronóstico, ya que al final esos son los inputs para el modelo, es lo primero que se necesita para poder simular después.




- Ver si puedo llegar con los modelos de pronósticos hechos, validados y con un modelo de optimización para la primera entrega.
Por ejemplo en los modelos de pronóstico nosotros tenemos métricas de error para ver que tan certero es el pronóstico, esos no son los KPI.



Ir probando los modelos y ver cuales aplican más a cada tipo de demanda.
- Pescar un producto de cada categoría para hacerlo representativo, y chantar todos los modelos en el gráfico y ver al ojo cuales se descartan si o si y cuales podrían ser útiles
- Encontrar la distribución de los datos, saber como distribuyen, que al final lo necesitamos como datos de entrada para el pronóstico
- Hacer un heatmap quizás del volumen de ventas y de las ganancias por cada producto, subcategoría y así. Ver la rentabilidad del producto

- Podría hacer el análisis del porcentaje de ganancias de cada uno de los 49 subgrupos de manera mensual y semanal. Es decir, para cada una de las semanas y para cada uno de los meses. Quizás podría hacer lo mismo con los 3 grupos grandes, no sé si vale la pena hacerlo con cada uno de los productos
- Hacer un análisis del stock que tienen hoy en día en términos de proporciones
    
## Cosas dichas por el profe

- Sacarle foto al sistema, demanda global, por producto, por subcategoría, subsubcategoría si es que hay.
- Ver si hay productos nuevos, productos que no se venden, productos obsoletos
- Costo del producto es el precio del proveedor. El costo es por día de almacenaje.
- El leadtime es el leadtime promedio en días, es el tiempo que pasa entre que yo pido el producto y cuando me llega.
- Costo por pedido es como un costo fijo por el producto
- 125 metros cubicos efectivos de la bodeg
- No siempre está el id del cliente, si lo tenemos, puede que tratemos de clusterizar,
ver relaciones entre productos de compra y clientes, tiene perros, gatos.
- Referencias son para punto de partida, ir más allá, por lo menos triplicar lo que nos envió
- Mostrar datos disponibles, hallazgos, saber como para donde va la cosa. Haber identificado que metodologías usar.
Ver modelos y técnicas de pronóstico que podríamos utilizar, tener la película clara de entender bien el problema,
como atacarlo, que metodologías tenemos.
    

## Librerías que pueden ser útiles

[Darts](https://unit8co.github.io/darts/)

[Prophet](https://facebook.github.io/prophet/docs/quick_start.html)

[Orbit](https://orbit-ml.readthedocs.io/en/latest/)

[tsfresh](https://tsfresh.readthedocs.io/en/latest/)

[AutoTS](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html)

[statsmodels](https://www.statsmodels.org/stable/index.html)

[featuretools](https://www.featuretools.com/)

[GluonTS](https://ts.gluon.ai/stable/)

[Pytorch-Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/)


## Reunión profesor 16 abril

- No podemos tener bola de cristal en el forecasting.
- Separar datos en entrenamiento y testeo
- Que productos consideramos


- Caso base
- Como proseguir
- En que nos deberíamos fijar ahora
- También podemos diseñar una política propia. Podemos cambiar una política
- Tenemos ue hacer cuadrar la política con nuestro problema
- Lo que no podemos hacer es asumir que el leadtime es cero
- No es válido que nos olvidemos de una restricción o consideración intrínsica del problema. Ahí
la política estaría definida frente a un problema diferente.
- Si la demanda cambia, la política podría funcionar mal. Lo que no puedo hacer es hacer una compra
y simular que el inventario me llegó al tiro.
- Podríamos cada cierto tiempo recalibrar parámetros de política o no recalibrarla y quedarme con lo que hice hace
6 meses.
- Modelo de optimización se puede pensar como que se calcula política, input son datos y output son decisiones.
Uno podría pensar que el modelo de optimización también sirve como una política. La política es como yo tomo
decisiones con los datos que tengo en el momento. Política es super general, probabilidades, sin probabilidades
podemos proponer políticas para resolver este problema.
- Forecast probabilístico
- XGBoost
- Descartaría el profe los modelos neuronales. Agregarlo al informe. Modelo de regresión lineal múltiple.
Le podemos meter feature engineering, para pasarsela a este modelo, podemos extraer muchas features que son relevantes
que le pueden servir para entregar mejor predicción. Como por ejemplo, definir vector de features el día de la semana
la semana del año, información sobre la tendencia, demanda acumulada, demanda del día anterior.
Si tenemos productos que tienen patrones complejos e información que no nos dieron, la podemos identificar
y pasarsela al modelo. Tomar modelos de pronóstico que ya identificamos, como baseline.
- Paper de Sinthetos, de demanda intermitente. Como clasificar los productos por patrón de demanda. Mirar productos
y separar productos por patrón de demanda. Explorar esto

- Tratar de etiquetar productos que no se venden, tenerlo identificado y a partir de ahí uno puede
sacar otras recomendaciones. Nosotros podemos definir nuestros criterios para determinar que productos están
activos. Centrar en datos que le vamos a poder hacer pronóstico. Hacer un diagnóstico de que es lo que hay
que hago, que stock tengo.

- Tratar de agrupar a los clientes, ver como las políticas impactan a los distintos clientes. Sobre todo si tenemos
clientes muy fieles y que queremos cuidar.

- Podríamos tratar desde ya implementar algunas políticas. Definir, aterrizar cual va a ser nuestro caso base. Diagrama
de como funciona nuestra política de caso base. Nuestra política, para tratar de definir como funciona todo.
- Empezar a probar cosas, quizás algún subset de productos, tomar último año de observaciones. Simular políticas
con proyecciones.

- Buen punto de partida, tratar de evaluar nuestras políticas propuestas con demanda histórica. Que es lo que yo
hubiera ejecutado, y que hubiera pasado. Ahora sería como replicar, ya que es determinístico. Ir registrando KPI,
quiebres de stock y ver como nos fue con una política.

- Tratar de ver lo antes posible, evaluar una política y ver indicadores de desempeño.
- Ver la cota de inferior perfecta, es una cota de rendimiento de política considerando que tenemos
toda la información futura. Si yo tuviera pronóstico perfecto, podríamos evaluar las políticas.
Ver como el máximo jugo que le podemos sacar al producto.

- Hartas cosas que podemos ver y profundizar. Hacer listado de cosas que podemos hacer, repartir pega y paralelizar.
- KPI, cantidad de compras

- Podríamos agregar el heatmap de las correlaciones. No es prioritario. Ponerle prioridad e interesante avanzar
en modelos de pronóstico, tomarlos, implementarlos y evaluarlos. Como dividir los datos, como se evalúan los
modelos, cuanto pronostican. Ver que quiero pronosticar. Tener ideas claras y ponerse a prototipar.

## Links útiles

Ver en Kaggle: Store Sales - Time Series Forecasting

[Darts Forecasting Deep Learning & Global Models](https://www.kaggle.com/code/ferdinandberr/darts-forecasting-deep-learning-global-models)

[Listen To Secrets in Your Data](https://www.kaggle.com/code/adnanshikh/listen-to-secrets-in-your-data)

[Darts Ensemble Store Sales Forecasting](https://www.kaggle.com/code/kelde9/darts-ensemble-stores-sales-forecasting#Machine-Learning-Model)

[Store Sales EDA, Prediction with TS](https://www.kaggle.com/code/kalilurrahman/store-sales-eda-prediction-with-ts)

[Store Sales](https://www.kaggle.com/code/hardikgarg03/store-sales)

[Exploring Time Series plots](https://www.kaggle.com/code/odins0n/exploring-time-series-plots-beginners-guide)

[Stores Sales Analysis Time Series](https://www.kaggle.com/code/kashishrastogi/store-sales-analysis-time-serie)

[Stores Sales TS Forecasting - A Comprehensive Guide](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide#9.-ACF-&-PACF-for-each-family)

[Customer Segmentation (K-Means) Analysis]()

[Customer Segmentation: Clustering](https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering#CLUSTERING)

[Customer Segmentation](https://www.kaggle.com/code/fabiendaniel/customer-segmentation)

[K-Means Clustering with Python](https://www.kaggle.com/code/prashant111/k-means-clustering-with-python)

[The Ultimate Step-by-Step Guide to Data Mining with PCA and KMeans](https://drlee.io/the-ultimate-step-by-step-guide-to-data-mining-with-pca-and-kmeans-83a2bcfdba7d)

[How could I find the indexes of data in each cluster after applying the PCA and a clustering method?](https://stackoverflow.com/questions/62626305/how-could-i-find-the-indexes-of-data-in-each-cluster-after-applying-the-pca-and)

[Understanding clusters after applying PCA then K-Means](https://datascience.stackexchange.com/questions/93100/understanding-clusters-after-applying-pca-then-k-means)

[Time Series in Python - Exponential Smoothing and Arima processes](https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788)

[K-Means clustering for mixed numeric and categorical data](https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data)

[K-Means, kmodes, and k-prototype](https://medium.com/@reddyyashu20/k-means-kmodes-and-k-prototype-76537d84a669)

[Unsupervised Learning using K-Prototype and DBScan](https://www.kaggle.com/code/rohanadagouda/unsupervised-learning-using-k-prototype-and-dbscan)

[Customer Segmentation in Python: A Practical Approach](https://www.kdnuggets.com/customer-segmentation-in-python-a-practical-approach)

[How to Build Customer Segmentation Models in Python](https://365datascience.com/tutorials/python-tutorials/build-customer-segmentation-models/)

[How to Perform Customer Segmentation in Python - ML Tutorial](https://www.freecodecamp.org/news/customer-segmentation-python-machine-learning/)

[Advanced Time Series Analysis](https://www.kaggle.com/code/bextuychiev/advanced-time-series-analysis-decomposition)

[Fourier Tranform for Time Series]()

[Demand Classification UK Retail](https://www.kaggle.com/code/danala26/demand-classification-uk-retail)

[Autoregressive Moving Average ARMA(p,q) Models for Time Series Analysis](https://www.quantstart.com/articles/Autoregressive-Moving-Average-ARMA-p-q-Models-for-Time-Series-Analysis-Part-3/#:~:text=Choosing%20the%20Best%20ARMA(p,achieved%2C%20for%20particular%20values%20of%20.)


[How does ACF & PACF identify the order of MA and AR terms?](https://stats.stackexchange.com/questions/281666/how-does-acf-pacf-identify-the-order-of-ma-and-ar-terms)

[Choosing the best q and p from ACF and PACF plots in ARMA-type modeling](https://www.baeldung.com/cs/acf-pacf-plots-arma-modeling)

[What are ACF and PACF Plots in Time Series Analysis?](https://ilyasbinsalih.medium.com/what-are-acf-and-pacf-plots-in-time-series-analysis-cb586b119c5d#:~:text=a%20time%20series.-,The%20ACF%20plot%20shows%20the%20correlation%20of%20a%20time%20series,%2C%20MA%2C%20and%20ARMA%20models.)

[Identifying the numbers of AR or MA terms in an ARIMA model](https://people.duke.edu/~rnau/411arim3.htm#plots)

[Autocorrelation and Time Series Methods](https://online.stat.psu.edu/stat462/node/188/)

[Demand Classification](https://frepple.com/blog/demand-classification/)

[https://datastud.dev/posts/python-seasonality-how-to](https://datastud.dev/posts/python-seasonality-how-to)

[How to cluster images based on visual similarity](https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34)

[Fourier Time Series](https://medium.com/@jcatankard_76170/forecasting-with-fourier-series-8196721e7a3a)

[XGboost](https://www.analyticsvidhya.com/blog/2024/01/xgboost-for-time-series-forecasting/#:~:text=XGBoost%2C%20short%20for%20Extreme%20Gradient,create%20a%20strong%20predictive%20model.)

[DeepAR](https://medium.com/@corymaklin/deepar-forecasting-algorithm-6555efa63444)

[Time Series RNN](https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/)

[Multivariate Time Series RNN](https://medium.com/@soubhikkhankary28/multivariate-time-series-forecasting-using-rnn-lstm-8d840f3f9aa7)

[Multivariate Time Series RNN forecasting](https://medium.com/@soubhikkhankary28/multivariate-time-series-forecasting-using-rnn-lstm-8d840f3f9aa7)
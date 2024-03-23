# Grupo Capstone - Gestión de Inventario

## Cosas por hacer e ideas

    - Hacer filtración de datos. Ver si quizás hay datos con magnitues que no corresponden. Revisar los outliers, que hacer con ellos.
    - Para los pronósticos se podria en algún momento intentar programar una RNN.
    - No olvidar el test estadístico. Ajustar los datos a una distribución y tener en cuenta
    su valor Chi-Cudadrado y el test de Kolmogorov-Smirnov. Lo que podría hacer es partir de lo más agregado a lo menos desagregado.
    Buscar tendencias para cada uno de los 3 grupos principales, encontrar una distribución para cada uno.
    Luego hacer lo mismo con cada uno de los 49 subgrupos, para que esto sea más preciso.
    Por último, hacerlo con cada uno de los 1800 productos individuales, revisar cuantas veces se ha vendido cada uno de ellos,
    si hay algunos que se dejaron de vender, cuando fue la última venta, ver si sacarlo si es que tiene pocos datos.
    - Ver como segmentar a los clientes, agruparlos. Ver también si hay ciertos productos que se compran en específico.
    - Ver si la demanda por producto es intermitente o no. Hay distintos modelos de pronóstico.
    - Ver si hay estacionareidad, si es que para cierta época ciertos productos se venden más o menos, cosas así.
    Por ejemplo, ver semanal/mensualmente la demanda de los medicamentos, ser capaz de que por ejemplo, quizás en junio
    se venden más respecto a todo el resto del año y de ahí desagregar por cada producto de medicamentos.
    - Ver si los datos tienen alguna pillería
    - Ver si puedo llegar con los modelos de pronósticos hechos, validados y con un modelo de optimización para la primera entrega.
    - Los KPI para la primera entrega van si o si, son las métricas de desempeño de la solución.
    Por ejemplo en los modelos de pronóstico nosotros tenemos métricas de error para ver que tan certero es el pronóstico, esos no son los KPI.
    Los KPI se asocian a la política de inventario, como una política es mejor que otra, bajo que criterios.
    - No es mala idea resolver el problema con OD, al menos en un sentido aproximado. Ver si podemos utilizar RL
    - Ver esto de los modelos ABC y Pareto
    - Graficar cosas desagregadas según distintos periodos de tiempo. Por tendencia mensual, semanal, diaria
    - La idea es agrupar los productos no tanto por temas de la cantidad demandada, sino por el tipo de demanda al que se enfrentan. Para poner ahí ahí los modelos de pronóstico
    - Una vez categorizados los productos lo que se puede hacer es buscar modelos de pronóstico por cada uno y ahí iterar para el que minimiza el error.
    - Fijarse en la tendencia, la idea es hacer el modelo de pronóstico de eso x semana, mientras más preciso mejor. Hacerlo por día nomás. Por precisión, idealmente sería mejor hacerlo producto a producto.
    - Categorizar la demanda, porque hay modelos que funcionan mejor que otros dependiendo del tipo de demanda.
    - Centrarse en como armar los modelos de pronóstico
    - Estudiar como funcionan los modelos de pronóstico, ya que al final esos son los inputs para el modelo,
    es lo primero que se necesita para poder simular después.
    - Modelos clásicos de time series forecasting son: Autoregression (AR), Moving Average (MA),
    Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA),
    Seasonal Autoregressive Integrated Moving Average (SARIMA),
    Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX),
    Vector Autoregression (VAR), Vector Autoregression Moving Average (VARMA),
    Vector Autoregression Moving Average with Exogenous Regressors (VARMAX),
    Simple Exponential Smoothing (SES), hOLT Winter's Exponential Smoothing (HWES).
    Ir probando los modelos y ver cuales aplican más a cada tipo de demanda.
    - Se podría minimizar una ponderación entre el MASE, el MES Y EL R2, para elegir.
    - Una vez tenemos los modelos de pronóstico, tenemos harto
    - Pescar un producto de cada categoría para hacerlo representativo, y chantar todos los modelos en el gráfico y ver al ojo cuales se descartan si o si y cuales podrían ser útiles
    - Ver si hay estacionariedad en las demandas porque en dicho caso, tendríamos que realizar un análisis por estación. Ver si tener modelos por tramos
    - Como presentar la información después, es importante eso.
    - Encontrar la distribución de los datos, saber como distribuyen, que al final lo necesitamos como datos de entrada para el pronóstico
    - Ver la importancia de lo que aporta en ganancia cada producto. Ver si por período de tiempo, o del total
    - Hacer un heatmap quizás del volumen de ventas y de las ganancias por cada producto, subcategoría y así. Ver la rentabilidad del producto
    - Se podría hacer un leadtime como variable aleatoria
    - Analizar harto, mientras más mejor
    - Para validar modelo de pronóstico, 75% datos para entrenar y 25% validar
    - Podría hacer el análisis del porcentaje de ganancias de cada uno de los 49 subgrupos de manera mensual y semanal. Es decir, para cada una de las semanas y para cada uno de los meses. Quizás podría
    hacer lo mismo con los 3 grupos grandes, no sé si vale la pena hacerlo con cada uno de los productos
    
    
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
[]()
[]()
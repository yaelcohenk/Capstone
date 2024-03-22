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
    
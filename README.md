# Capstone 

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

### Data analysis
- Perform data filtering. Check if there are values with very different magnitudes. Review the outliers and decide what to do with them.
- Check if the data has negative values or nulls.
- See how to segment the customers, group them. Also check if there are specific products that are being purchased.
- Analyze whether the demand for each product is intermittent or not. There are different forecasting models for that.
- Check for seasonality, i.e., if certain products sell more or less depending on the time of year. For example, analyze the weekly/monthly demand for medications, and be able to detect if, say, in June they sell more compared to the rest of the year — and then break that down by product.
- You could analyze the profit percentage of each of the 49 subgroups on a monthly and weekly basis. Maybe you could do the same with the 3 larger groups — not sure if it’s worth doing it for each individual product.
- Check for seasonality in the demand because if that’s the case, we’d need to carry out an analysis by season. See if it makes sense to create models by time segments.
- Plot disaggregated data based on different time periods. Look at monthly, weekly, daily trends.
- Pay attention to the trend. The idea is to build a forecasting model by week — the more accurate, the better. But do it daily. For accuracy, ideally it would be better to do it product by product.
- Evaluate the importance of the contribution to profit for each product — either by time period or overall.
- You could build a correlation matrix by year-month to see which products tend to be sold together.

### Forecasting models

- Use RNN for forecasting (LSTM).
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

### Statistical validation of the data

- Don’t forget the statistical tests. Fit the data to a distribution and consider the Chi-Square value and the Kolmogorov-Smirnov test. Then, the same could be done for each of the 49 subgroups to make the analysis more precise. Eventually, it could be done for each individual product as well.

### Tips

- One idea could be to group the products not by the quantity demanded, but by the type of demand they face. This would help in choosing the right forecasting model later on.
- Categorize the demand, because some models work better than others depending on the demand type.
- Look into ABC and Pareto models.
- Start from the most aggregated level and move toward the more disaggregated.
- Review each product, check how many times each one has been sold, if there are some that have stopped selling, when the last sale was — and consider removing them if there’s very little data.
- Lead time could be modeled as a random variable.
- To validate the forecasting model, use 75% of the data for training and 25% for validation.
- Think about how to present the information — that's important.
- Focus on how to build the forecasting models.
- KPIs are mandatory for the first delivery; they’re the performance metrics of the solution.
- It’s not a bad idea to try solving the problem with Operations Research (OR), at least approximately. Consider whether Reinforcement Learning (RL) can be used.
- The KPIs are tied to the inventory policy — showing how one policy is better than another, under which criteria.
- Once the products are categorized, you can find forecasting models for each category, and then iterate to find the one that minimizes the error.
- You could minimize a weighted combination of MASE, MES, and R² to select the best model.
- Study how forecasting models work — since they’re the input for the overall model, it’s the first thing needed before simulation.
- See if we can have the forecasting models completed, validated, and with an optimization model ready for the first delivery.
- For example, in the forecasting models we have error metrics to assess how accurate the forecast is — but those are not the KPIs.
- Start testing the models and see which ones apply best to each type of demand.
- Take one product from each category to use as a representative example, apply all the models to it, plot them all on a graph, and visually discard the ones that clearly don’t work and identify the ones that might be useful.
- Find the data distribution — understand how the data is distributed, since that will be needed as input for the forecasting.
- Maybe create a heatmap of sales volume and profits by product, subcategory, etc. Analyze product profitability.
- You could analyze the profit percentage of each of the 49 subgroups on a monthly and weekly basis. That is, for each week and each month. Maybe do the same with the 3 larger groups — not sure if it’s worth doing for every single product.
- Do an analysis of the current stock in terms of proportions.
  
## Professor tips

- Take a snapshot of the system: global demand, by product, by subcategory, and sub-subcategory if it exists.  
- Check if there are new products, unsold products, obsolete products.  
- The product cost is the supplier’s price. The holding cost is calculated per day of storage.  
- Lead time is the average lead time in days — the time between when I order the product and when it arrives.  
- Ordering cost is like a fixed cost per product.  
- The warehouse has 125 cubic meters of effective storage space.  
- The customer ID is not always available — if we have it, we might try to cluster and identify relationships between purchased products and customers (e.g., whether they have dogs, cats, etc.).  
- References are just a starting point — aim to go beyond them, at least triple what we were given.  
- Present the available data, findings, and show where things are heading. Be clear on what methodologies we’ll use.  
Understand which forecasting models and techniques we could apply, and have a clear understanding of the problem, how to tackle it, and which methods we have at our disposal.
    

## Useful libraries

[Darts](https://unit8co.github.io/darts/)

[Prophet](https://facebook.github.io/prophet/docs/quick_start.html)

[Orbit](https://orbit-ml.readthedocs.io/en/latest/)

[tsfresh](https://tsfresh.readthedocs.io/en/latest/)

[AutoTS](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html)

[statsmodels](https://www.statsmodels.org/stable/index.html)

[featuretools](https://www.featuretools.com/)

[GluonTS](https://ts.gluon.ai/stable/)

[Pytorch-Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/)

## Links

Kaggle: Store Sales - Time Series Forecasting

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

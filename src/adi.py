import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
productos_vigentes = pd.read_excel(os.path.join("datos", "prod_vigentes.xlsx"))
productos_vigentes = productos_vigentes["Descripción"].to_list()

ventas_diarias_prod = pd.read_excel(os.path.join("datos", "ventas_diarias_productos.xlsx"))
ventas_diarias_prod = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin(productos_vigentes)]

productos_unicos = ventas_diarias_prod["Descripción"].unique().tolist()

listas = list()
repti_resercoir = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin([productos_unicos[0]])]
repti_resercoir = repti_resercoir.sort_values(by="Fecha")
repti_resercoir["diferencia_tiempo"] = repti_resercoir["Fecha"].diff()

adi = repti_resercoir["diferencia_tiempo"].mean()
# print(repti_resercoir)
# print(adi.days)


for producto in productos_unicos:
    producto_db = ventas_diarias_prod[ventas_diarias_prod["Descripción"].isin([producto])]
    producto_db = producto_db.sort_values(by="Fecha")
    producto_db["diferencia_tiempo"] = producto_db["Fecha"].diff()
    adi = producto_db["diferencia_tiempo"].mean().days
    
    cv_sqr = (producto_db["Cantidad"].std() / producto_db["Cantidad"].mean()) ** 2  
    listas.append([producto, adi, cv_sqr])


dataset = pd.DataFrame(listas, columns=["Descripción", "ADI", "CV2"])
print(dataset)
dataset.to_excel("adi_fco.xlsx", index=False)


## Defining a fuction for categorization

def category(df):
    a = 0
    
    if((df['ADI']<=1.34) & (df['CV2']<=0.49)):
        a='Smooth'
    if((df['ADI']>=1.34) & (df['CV2']>=0.49)):  
        a='Lumpy'
    if((df['ADI']<1.34) & (df['CV2']>0.49)):
        a='Erratic'
    if((df['ADI']>1.34) & (df['CV2']<0.49)):
        a='Intermittent'
    return a

dataset["categoria"] = dataset.apply(category, axis=1)
sns.scatterplot(x='CV2',y='ADI',hue='categoria',data=dataset)
plt.show()

print(dataset.categoria.value_counts())


sys.exit()

productos_agrupados = ventas_diarias_prod.groupby(['Descripción','Fecha']).agg(total_sale=('Cantidad','sum')).reset_index()
cv_data = productos_agrupados.groupby('Descripción').agg(average=('total_sale','mean'),
                                                    sd=('total_sale','std')).reset_index()


cv_data['cv_sqr'] = (cv_data['sd'] / cv_data['average']) ** 2
print(productos_agrupados)
print(cv_data)

prod_by_date= ventas_diarias_prod.groupby(['Descripción','Fecha']).agg(count=('Descripción','count')).reset_index()

skus= prod_by_date["Descripción"].value_counts()
print(skus)

new_df= pd.DataFrame()

for i in range(len(skus.index)):
    a= prod_by_date[prod_by_date['Descripción']==skus.index[i]]
    a['previous_date']=a['Fecha'].shift(1)
    new_df=pd.concat([new_df,a],axis=0)
# print(productos_vigentes)

new_df.info()

new_df['duration']=new_df['Fecha']- new_df['previous_date']
new_df['Duration']=new_df['duration'].astype(str).str.replace('days','')
new_df['Duration']=pd.to_numeric(new_df['Duration'],errors='coerce')
## Calculating ADI

ADI = new_df.groupby('Descripción').agg(ADI = ('Duration','mean')).reset_index()
print(ADI)

adi_cv=pd.merge(ADI,cv_data)

adi_cv.to_excel("adi_kaggle.xlsx", index=False)

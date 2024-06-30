import itertools

from fieles.cf import clientes_fieles
from fieles.diccionario_prod_fieles import obtener_suma_cantidad_por_item_id_ordenado
from fieles.prod_clientes_fieles import productos_comprados_por_clientes_fieles

def obtener_lista_productos_fieles(df_clientes, datos_items, obtener_suma_cantidad_por_item_id_ordenado, productos_comprados_por_clientes_fieles):
    
    suma_cantidad_por_item_id_ordenado = obtener_suma_cantidad_por_item_id_ordenado(df_clientes, productos_comprados_por_clientes_fieles)

    # Crear una lista de listas de clientes fieles por año
    lista_lista_clientes_fieles = []
    for i in range(2020, 2024):
        clientes = clientes_fieles(df_clientes, i)
        lista_lista_clientes_fieles.append(clientes)
    
    # Unir todas las listas de clientes fieles en una sola lista
    lista_clientes_fieles = list(itertools.chain(*lista_lista_clientes_fieles))

    # Obtener los IDs de los productos fieles
    ids = list(suma_cantidad_por_item_id_ordenado.keys())

    # Filtrar el dataframe de clientes para obtener solo los productos fieles
    df_productos_fieles = df_clientes[df_clientes['item_id'].isin(ids)]

    # Eliminar duplicados basados en 'item_id'
    df_productos_fieles = df_productos_fieles.drop_duplicates(subset=['item_id'])

    # Obtener los IDs únicos de los productos fieles
    ids_productos_fieles = df_productos_fieles['item_id'].unique()

    # Filtrar los datos de items para obtener solo los productos fieles
    filtro = datos_items['item_id'].isin(ids_productos_fieles)
    productos_fieles_dict = dict(datos_items[filtro][['item_id', 'description']].values)

    # Convertir los valores del diccionario a una lista
    lista_productos_fieles = list(productos_fieles_dict.values())
    
    return lista_productos_fieles

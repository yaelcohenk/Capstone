from fieles.cf import clientes_fieles

def productos_comprados_por_clientes_fieles(df_clientes, anio):

    df_anio = df_clientes[df_clientes['date'].dt.year == anio]
    clientes_fieles_anio = clientes_fieles(df_anio, anio)
    df_clientes_fieles_anio = df_anio[df_anio['client_id'].isin(clientes_fieles_anio)]
    productos_comprados = df_clientes_fieles_anio.groupby('item_id')['quantity'].sum().reset_index()
    productos_comprados = productos_comprados.sort_values(by='quantity', ascending=False)

    return productos_comprados
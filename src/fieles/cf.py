def clientes_fieles(df_clientes, anio):
    df_anio = df_clientes[df_clientes['date'].dt.year == anio]

    if anio == 2024:
        fecha_mas_reciente = df_anio['date'].max()
        mes_anterior = fecha_mas_reciente.month - 1
        df_anio = df_anio[df_anio['date'].dt.month <= mes_anterior]

    compras_por_cliente_mes = df_anio.groupby(['client_id', df_anio['date'].dt.month]).size().reset_index(name='compras')
    primer_mes_compra = df_anio.groupby('client_id')['date'].min().dt.month

    clientes_fieles = []
    for client_id, primer_mes in primer_mes_compra.items():
        if anio == 2024:
            meses_esperados = set(range(primer_mes, mes_anterior + 1))
        else:
            meses_esperados = set(range(primer_mes, 13))

        meses_compra_cliente = set(compras_por_cliente_mes[compras_por_cliente_mes['client_id'] == client_id]['date'])

        # Verificar si la primera compra es en diciembre y el año es diferente al año actual
        if primer_mes == 12 and df_clientes[df_clientes['client_id'] == client_id]['date'].dt.year.min() != anio:
            continue

        if meses_esperados.issubset(meses_compra_cliente):
            clientes_fieles.append(client_id)

    return clientes_fieles
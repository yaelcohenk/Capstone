def dataframe_from_value_counts(dataframe, columns):
    datos = dataframe.value_counts().reset_index()
    datos.columns = columns
    return datos

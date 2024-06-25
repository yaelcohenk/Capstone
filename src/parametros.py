import os

PATH_SALES_DATA = os.path.join("datos", "data_sales.csv")
PATH_STOCK_DATA = os.path.join("datos", "data_items.csv")


PATH_VENTAS_ITEM_IDS = os.path.join("plots", "eda_ventas", "ventas_item_ids.png")
PATH_GRUPOS_INVENTARIO = os.path.join("plots", "eda_inventario", "grupos_inventario.png")
PATH_SUBGRUPOS_INVENTARIO = os.path.join("plots", "eda_inventario", "subgrupos_inventario.png")

PATH_VENTAS_ANUALES_GRUPO = os.path.join("plots", "eda_conjunto", "ventas_anuales_grupos.png")
PATH_SUBGRUPOS_MAS_VENDIDOS = os.path.join("plots", "eda_conjunto", "elementos_mas_vendidos_subgrupos.png")

PATH_VENTAS_DIARIAS_GRUPO_DATA = os.path.join("datos", "ventas_diarias_grupos.xlsx")
PATH_VENTAS_DIARIAS_SUBGRUPO_DATA = os.path.join("datos", "ventas_diarias_subgrupos.xlsx")
PATH_VENTAS_DIARIAS_PRODUCTOS_DATA = os.path.join("datos", "ventas_diarias_productos.xlsx")

PATH_VENTAS_SEMANALES_GRUPO_DATA = os.path.join("datos", "ventas_semanales_grupos.xlsx")
PATH_VENTAS_SEMANALES_SUBGRUPO_DATA = os.path.join("datos", "ventas_semanales_subgrupos.xlsx")
PATH_VENTAS_SEMANALES_PRODUCTOS_DATA = os.path.join("datos", "ventas_semanales_productos.xlsx")
PATH_ADI_PRODUCTOS = os.path.join("datos", "adi_productos.xlsx")
PATH_METRICAS_CROSTON = os.path.join("datos", "metricas_croston.xlsx")
PATH_METRICAS_XGBOOST = os.path.join("datos", "metricas_xgboost.xlsx")
PATH_METRICAS_PROPHET = os.path.join("datos", "metricas_prophet.xlsx")
PATH_METRICAS_HOLTWINTER = os.path.join("datos", "metricas_holt_winters.xlsx")



PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS = os.path.join("datos", "ventas_diarias_productos_vigentes_no_outliers.xlsx")
PATH_VENTAS_PRODUCTOS_VIGENTES_NO_OUTLIERS_W_FEATURES = os.path.join("datos",
                                                                      "ventas_diarias_prod_vigentes_no_outliers_w_features.xlsx")


# Parámetros para XGBoost

FEATURES = ['dayofyear',
            'dayofweek',
            'quarter',
            'month',
            'year',
            'weekofyear',
            "diff_tiempo_venta",
            "fourier_transform",
            "trend",
            "seasonal",
            "ventas_fecha_anterior"]
    
TARGET = "Cantidad"
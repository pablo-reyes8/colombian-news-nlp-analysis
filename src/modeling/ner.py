import pandas as pd 
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def interactive_ner_plot(df):
    """
    Crea un panel interactivo (compatible con Google Colab) para explorar entidades reconocidas (NER)
    con filtros por categoría y fecha, y selección del top-N por frecuencia.

    El panel muestra cuatro gráficos de barras horizontales (PER, ORG, LOC, MISC) en una misma figura.
    Cada barra representa el conteo de ocurrencias por texto de entidad.

    Args:
        df (pandas.DataFrame): DataFrame con, al menos, las columnas:
            - 'categoria_grupo' (str): Categoría usada para filtrar.
            - 'published_date' (str o datetime-like): Fecha de publicación.
            - 'text' (str): Texto de la entidad.
            - 'label' (str): Etiqueta NER en {"PER", "ORG", "LOC", "MISC"}.
            - 'url' (str): Usada para el conteo por grupo (se cuenta el número de filas por (text, label)).

    Returns:
        None: Renderiza en la notebook los widgets y el gráfico interactivo (Plotly).

    Notas:
        - Incluye tres widgets: categoría, fecha (con opción 'Todos') y top-N de palabras.
        - La lista de fechas del selector se construye filtrando previamente por
          `published_date > '2025-07-31'`. Ajuste esta condición si requiere otro horizonte temporal.
        - Requiere `ipywidgets`, `plotly` y un entorno que soporte su visualización (Colab o Jupyter).
        - Para mostrar el UI se utiliza `display` (de `IPython.display`).
        - El conteo se calcula como número de filas por combinación ('text', 'label') usando `url` como
          columna a contar (`.count()`); si prefiere conteo de ocurrencias únicas, adapte la agregación.
    """
    category_widget = widgets.Dropdown(
        options=['Todos'] + sorted(list(df['categoria_grupo'].unique())),
        description='Categoria:',
        value='Todos'
    )

    date_options_ls = sorted(df[pd.to_datetime(df['published_date']) > pd.to_datetime('2025-07-31')]['published_date'].unique())
    date_widget = widgets.Dropdown(
        options=['Todos'] + date_options_ls,
        description='Fecha:',
        value='Todos',
    )

    top_n_widget = widgets.Dropdown(
        options=range(1, 25),
        description='Number of words:',
        value=10,
    )

    def update_and_plot(category, date, top_n):

        filtered_df = df.copy()

        if category != 'Todos':
            filtered_df = filtered_df[filtered_df['categoria_grupo'] == category]

        if date != 'Todos':
            filtered_df = filtered_df[filtered_df['published_date'] == date]

        grouped_data = filtered_df.groupby(['text', 'label'])['url'].count().reset_index()

        fig = make_subplots(
            rows=1,
            cols=4,
            shared_yaxes=False,
            subplot_titles=("PER", "ORG", "LOC", "MISC"),
            horizontal_spacing=0.1,
        )

        for i, label in enumerate(["PER", "ORG", "LOC", "MISC"], 1):
            data_subset = grouped_data[grouped_data['label'] == label].sort_values('url', ascending=False).head(top_n)

            fig.add_trace(
                go.Bar(
                    x=data_subset['url'],
                    y=data_subset['text'],
                    name=label,
                    orientation='h',
                    marker=dict(cornerradius=10)
                ),
                row=1, col=i
            )

        fig.update_layout(
            height=600,
            width=1200,
            title_text="Interactive NER Plot",
            showlegend=False,
            yaxis_autorange='reversed',
        )

        fig.show()

    out = widgets.interactive_output(
        update_and_plot,
        {'category': category_widget, 'date': date_widget, 'top_n': top_n_widget}
    )

    ui = widgets.HBox([category_widget, date_widget, top_n_widget])
    display(ui, out)
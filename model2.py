import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import indicative_func 
import price_combined 
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

def cargar_datos(archivo_csv):
    """
    Carga los datos desde un archivo CSV.
    
    Args:
        archivo_csv: Ruta al archivo CSV con los datos de destinos
        
    Returns:
        DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(archivo_csv)
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def normalizar_caracteristicas(df):
    """
    Normaliza las características binarias para que tengan valores entre 0 y 1.
    
    Args:
        df: DataFrame con las características binarias
        
    Returns:
        DataFrame con características normalizadas
    """
    columnas_binarias = [
        'Arquitectura_Monuments', 'Art_Cultura', 'Gastronomia_Begudes',
        'Historia_Tradicio', 'Naturalesa', 'Platja', 'Events_Festvals',
        'Compres', 'Nightlife_Entreteniment', 'Ciutat_EstructuraUrbana',
        'Especific'
    ]
    
    # Verificar qué columnas están disponibles en el dataframe
    columnas_disponibles = [col for col in columnas_binarias if col in df.columns]
    
    # Normalizar columnas disponibles
    for col in columnas_disponibles:
        if col in df.columns:
            df[col] = df[col] / 5  # Normalizar valores (5 -> 1.0, 2 -> 0.4, 0 -> 0.0)
    
    return df

def obtener_nombre_categoria(clave):
    """
    Obtiene el nombre legible de una categoría a partir de su clave.
    
    Args:
        clave: Clave de la categoría en el sistema
        
    Returns:
        Nombre legible de la categoría
    """
    categorias = {
        'Arquitectura_Monuments': 'Arquitectura y monumentos',
        'Art_Cultura': 'Arte y cultura',
        'Gastronomia_Begudes': 'Gastronomía y bebidas',
        'Historia_Tradicio': 'Historia y tradición',
        'Naturalesa': 'Naturaleza',
        'Platja': 'Playa',
        'Events_Festvals': 'Eventos y festivales',
        'Compres': 'Compras',
        'Nightlife_Entreteniment': 'Vida nocturna y entretenimiento',
        'Ciutat_EstructuraUrbana': 'Ciudad y estructura urbana',
        'Especific': 'Elementos específicos'
    }
    return categorias.get(clave, clave)

def obtener_preferencias_usuario():
    """
    Solicita al usuario sus preferencias para cada categoría.
    
    Returns:
        Diccionario con las preferencias del usuario
    """
    categorias = {
        'Arquitectura_Monuments': 'Arquitectura y monumentos',
        'Art_Cultura': 'Arte y cultura',
        'Gastronomia_Begudes': 'Gastronomía y bebidas',
        'Historia_Tradicio': 'Historia y tradición',
        'Naturalesa': 'Naturaleza',
        'Platja': 'Playa',
        'Events_Festvals': 'Eventos y festivales',
        'Compres': 'Compras',
        'Nightlife_Entreteniment': 'Vida nocturna y entretenimiento',
        'Ciutat_EstructuraUrbana': 'Ciudad y estructura urbana',
        'Especific': 'Elementos específicos'
    }
    
    preferencias = {}
    print("\n--- SISTEMA DE RECOMENDACIÓN DE DESTINOS TURÍSTICOS ---")
    print("Para cada categoría, indique su preferencia:")
    print("5: Me interesa mucho, 2: Neutral, 0: No me interesa")
    
    for key, descripcion in categorias.items():
        while True:
            try:
                valor = int(input(f"{descripcion}: "))
                if valor in [0, 2, 5]:
                    preferencias[key] = valor / 5  # Normalizar
                    break
                else:
                    print("Por favor, ingrese 0, 2 o 5.")
            except ValueError:
                print("Por favor, ingrese un número válido.")
    
    return preferencias

def calcular_contribucion_categorias(preferencias_usuario, destino_vector, columnas_preferencias):
    """
    Calcula la contribución de cada categoría a la puntuación final.
    
    Args:
        preferencias_usuario: Vector de preferencias del usuario
        destino_vector: Vector de características del destino
        columnas_preferencias: Lista de nombres de columnas de preferencias
    
    Returns:
        Lista de tuplas (categoría, contribución, valor_usuario, valor_destino)
    """
    contribuciones = []
    
    # Normalizar los vectores para calcular las contribuciones correctamente
    norma_usuario = np.linalg.norm(preferencias_usuario)
    norma_destino = np.linalg.norm(destino_vector)
    
    if norma_usuario == 0 or norma_destino == 0:
        return contribuciones
    
    for i, columna in enumerate(columnas_preferencias):
        # Contribución de esta categoría al producto escalar
        contribucion = (preferencias_usuario[i] * destino_vector[i]) / (norma_usuario * norma_destino)
        
        # Solo considerar contribuciones positivas
        if contribucion > 0:
            contribuciones.append((columna, contribucion, preferencias_usuario[i], destino_vector[i]))
    
    # Ordenar contribuciones de mayor a menor
    contribuciones.sort(key=lambda x: x[1], reverse=True)
    return contribuciones

def generar_explicacion(destino, contribuciones, puntuacion, df):
    """
    Genera una explicación personalizada para un destino recomendado.
    
    Args:
        destino: Nombre del destino
        contribuciones: Lista de contribuciones por categoría
        puntuacion: Puntuación general del destino
        df: DataFrame con datos completos
    
    Returns:
        Explicación textual del destino
    """    
    # Filtrar el dataframe para este destino
    fila_destino = df[df['Destinations'] == destino].iloc[0]
    
    # Añadir las principales categorías que contribuyen a la recomendación
    explicacion = "Este destino destaca especialmente en:\n"
    
    for i, (categoria, contrib, pref_usuario, valor_destino) in enumerate(contribuciones[:3]):
        nombre_cat = obtener_nombre_categoria(categoria)
        
        # Convertir valores normalizados de vuelta a escala original para mejor comprensión
        valor_usuario_escala = int(pref_usuario * 5)
        valor_destino_escala = int(valor_destino * 5)
        
        # Añadir explicación específica basada en la categoría
        if valor_destino_escala >= 4:  # Destino muy bueno en esta categoría
            if valor_usuario_escala >= 4:  # Al usuario le interesa mucho
                explicacion += f"- {nombre_cat}: Este destino es excelente en esta categoría que tanto te interesa.\n"
            else:  # Al usuario le interesa moderadamente
                explicacion += f"- {nombre_cat}: Aunque no es tu prioridad principal, este destino ofrece experiencias destacables.\n"
        else:  # Destino moderadamente bueno
            if valor_usuario_escala >= 4:  # Al usuario le interesa mucho
                explicacion += f"- {nombre_cat}: Ofrece buenas experiencias en esta categoría que es importante para ti.\n"
            else:  # Al usuario le interesa moderadamente
                explicacion += f"- {nombre_cat}: Hay opciones interesantes que coinciden con tus preferencias moderadas.\n"
    
    # Añadir información adicional si está disponible
    if 'Description_x' in fila_destino and pd.notna(fila_destino['Description_x']):
        explicacion += f"\nDescripción: {fila_destino['Description_x']}\n"
    
    if 'Best_Time_to_Travel' in fila_destino and pd.notna(fila_destino['Best_Time_to_Travel']):
        explicacion += f"Mejor época para viajar: {fila_destino['Best_Time_to_Travel']}\n"
    
    return explicacion

def recomendar_destinos_explicativo(df, preferencias_usuario, n_recomendaciones=5):
    """
    Recomienda destinos basados en las preferencias del usuario y genera explicaciones.
    
    Args:
        df: DataFrame con los datos de destinos
        preferencias_usuario: Diccionario con las preferencias del usuario
        n_recomendaciones: Número de destinos a recomendar
        
    Returns:
        DataFrame con los destinos recomendados, sus puntuaciones y explicaciones
    """
    # Seleccionar solo columnas disponibles en las preferencias
    columnas_preferencias = [col for col in preferencias_usuario.keys() if col in df.columns]
    
    # Si no hay columnas de preferencia disponibles, devolver mensaje de error
    if not columnas_preferencias:
        print("No hay coincidencias entre las preferencias del usuario y las columnas del DataFrame.")
        return None
    
    # Crear vector de preferencias del usuario
    vector_preferencias = np.array([preferencias_usuario.get(col, 0) for col in columnas_preferencias])
    usuario_vector = vector_preferencias.reshape(1, -1)
    
    # Crear matriz de características de destinos
    destinos_matriz = df[columnas_preferencias].values
    
    # Calcular similitud coseno entre preferencias del usuario y destinos
    similitudes = cosine_similarity(usuario_vector, destinos_matriz)[0]
    
    # Asignar puntuaciones a destinos
    df_resultados = df.copy()
    df_resultados['puntuacion'] = similitudes
    
    # Ordenar por puntuación descendente
    df_recomendaciones = df_resultados.sort_values(by='puntuacion', ascending=False).head(n_recomendaciones)
    
    # Añadir explicaciones
    explicaciones = []
    for idx, row in df_recomendaciones.iterrows():
        destino = row['Destinations']
        destino_vector = df_resultados.loc[idx, columnas_preferencias].values
        
        # Calcular contribuciones de cada categoría
        contribuciones = calcular_contribucion_categorias(
            vector_preferencias, 
            destino_vector, 
            columnas_preferencias
        )
        
        # Generar explicación para este destino
        explicacion = generar_explicacion(destino, contribuciones, row['puntuacion'], df)
        explicaciones.append(explicacion)
    
    # Añadir explicaciones al DataFrame
    df_recomendaciones['explicacion'] = explicaciones
    
    return df_recomendaciones[['Destinations', 'Country_x', 'puntuacion', 'Description_x', 'Best_Time_to_Travel', 'explicacion']]

def mostrar_recomendaciones_explicativas(origin, recomendaciones):
    """
    Muestra las recomendaciones de destinos al usuario con explicaciones.
    
    Args:
        recomendaciones: DataFrame con los destinos recomendados y explicaciones
    """
    if recomendaciones is None or recomendaciones.empty:
        print("No se encontraron recomendaciones.")
        return
    
    print("\n--- DESTINOS RECOMENDADOS ---")
    millor_preu = 1000000
    millor_aero = None
    for idx, row in recomendaciones.iterrows():
        print(f"\n {row['Destinations']}, {row['Country_x']}")
        print(f"Puntuación de coincidencia: {row['puntuacion']:.2f} ({row['puntuacion']:.0%})")
        print("\n" + row['explicacion'])
        print("-" * 50)
        print(row['Destinations'])
        aeros = price_combined.get_airports_by_city(row['Destinations'])
        for aero in aeros:
            nou = indicative_func.main([origin], [aero['IATA']], 2025, 10)
            tornada = indicative_func.main([aero['IATA']], [origin], 2025, 10)
            if nou != 'No prices available':
                if nou[0] + tornada[0]< millor_preu:
                    millor_preu = nou[0] + tornada[0]
                    millor_aero = aero['IATA']
        
        print(f"El mejor precio es {millor_preu} y el aeropuerto de destino es {millor_aero} (ida y vuelta)")

        location_id = price_combined.get_airport_id_from_iata(millor_aero)
        api_key = 'XXXXXXXXXXXXXx'
        cars_data = price_combined.search_car_hire(api_key, location_id, {"year": 2025, "month": 5, "day": 15}, {"year": 2025, "month": 5, "day": 19})
        if cars_data:
            car_hire_details = price_combined.extract_aggregate_total_average(cars_data)
        
        for car in car_hire_details:
            print(f"  Vehicle Type: {car['vehicleType']}")
            print(f"  Seats: {car['seats']}, Bags: {car['bags']}")
            print(f"  Total Average Price: {car['totalAveragePrice']}")

def adaptar_columnas(df):
    """
    Adapta los nombres de las columnas si es necesario.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        DataFrame con columnas adaptadas
    """
    # Mapeo de posibles variaciones en nombres de columnas
    mapeo_columnas = {
        'Arquitectura_Monuments': 'Arquitectura_Monuments',
        'Arquitectura Monuments': 'Arquitectura_Monuments',
        'ArquitecturaMonuments': 'Arquitectura_Monuments',
        'Art_Cultura': 'Art_Cultura',
        'Art Cultura': 'Art_Cultura',
        'ArtCultura': 'Art_Cultura',
        # Añadir más mapeos según sea necesario
    }
    
    # Renombrar columnas según el mapeo
    for col in df.columns:
        for key, value in mapeo_columnas.items():
            if col.lower().replace("_", "").replace(" ", "") == key.lower().replace("_", "").replace(" ", ""):
                df = df.rename(columns={col: value})
    
    return df

def main(origin):
        # Cargar datos
    archivo = "db.csv"  
    df = cargar_datos(archivo)
    
    # Adaptar columnas si es necesario
    df = adaptar_columnas(df)
    
    # Normalizar características
    df = normalizar_caracteristicas(df)
    
    # Obtener preferencias del usuario
    preferencias = obtener_preferencias_usuario()
    
    # Recomendar destinos con explicaciones
    recomendaciones = recomendar_destinos_explicativo(df, preferencias)
    # Mostrar recomendaciones explicativas
    mostrar_recomendaciones_explicativas(origin, recomendaciones)

def get_country_code(city_name, max_retries=3):

    geolocator = Nominatim(user_agent="city_country_lookup_app")

    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(city_name, language='en', addressdetails=True, timeout=10)
            if location:
                address = location.raw.get('address', {})
                country_code = address.get('country_code')
                if country_code:
                    return country_code.upper()
            break 
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            break
    return None

if __name__ == "__main__":
    ciutat = input("Introduce la ciudad de origen  (ej. Barcelona): ")
    aeros = price_combined.get_airports_by_city(ciutat)
    if not aeros:
        print("No se encontraron aeropuertos para la ciudad proporcionada.")
    else:
        print(aeros)

    aero = input("Introduce el IATA del aeropuerto de origen: ")
    main(aero)



# ==============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import os
import time
import pandas as pd
import numpy as np
# Flask y folium para la app web
from flask import Flask, render_template, request, jsonify
import folium
from folium import FeatureGroup
# Sklearn provee modelos y utilidades ML
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KDTree
import warnings

# Ignorar advertencias no criticas para limpiar consola
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ==============================================================================
# CONFIGURACIÓN Y VARIABLES GLOBALES
# ==============================================================================

# Dataset path de Kraggle 
DATASET_FILE = "US_Accidents_March23.csv"

# Coordenadas centrales de las ciudades a gestionar
CIUDADES = {
    "Los Angeles": [34.0522, -118.2437],
    "New York": [40.7128, -74.0060],
    "Chicago": [41.8781, -87.6298],
    "Miami": [25.7617, -80.1918],
    "Houston": [29.7604, -95.3698]
}

# Estado global para mantener el modelo y sus componentes
modelo_actual = {
    "ciudad": None,
    "mlp": None,
    "scaler": None,
    "encoder": None,
    "kdtree": None,
    "df": None,        # DataFrame histórico para consultas
    "grid_steps": None # Pasos lat/lng para calcular bounding box
}

# ==============================================================================
# LÓGICA DE MACHINE LEARNING
# ==============================================================================

def preparar_dataset_probabilistico(df_reales):
    # Preparar dataset
    # Complementar con negativos temporales y espaciales (baja probabilidad / ruido)
    
    # POSITIVOS (Accidentes Reales) -> Probabilidad = 1.0
    df_pos = df_reales.copy()
    df_pos['Probabilidad'] = 1.0 

    # Ruido para distribuir puntos
    # Añadimos un pequeño desplazamiento aleatorio (aprox 500m) a los puntos reales.
    # Se evita atribuir a un punto exacto el riesgo 100%, generalizando mejor por zonas
    ruido_lat = np.random.normal(0, 0.006, len(df_pos))
    ruido_lng = np.random.normal(0, 0.006, len(df_pos))
    df_pos['Start_Lat'] = df_pos['Start_Lat'] + ruido_lat
    df_pos['Start_Lng'] = df_pos['Start_Lng'] + ruido_lng

    # Negativos temporales -> Probabilidad = 0.2
    # Se generan puntos con las mismas coordenadas pero en horas y condiciones climáticas distintas.
    # Esto para enseñar al modelo que en otros momentos esos puntos son seguros.
    # Dado que el dataset comprende unicamente accidentes, estos negativos son "suaves".
    df_neg_temporal = df_pos.copy()
    df_neg_temporal['Hour'] = np.random.randint(0, 24, len(df_pos))
    codigos_clima = df_pos['Weather_Code'].unique()
    df_neg_temporal['Weather_Code'] = np.random.choice(codigos_clima, len(df_pos))
    df_neg_temporal['Probabilidad'] = 0.2 

    # Ruido espacial -> Probabilidad = 0.0
    # Puntos completamente aleatorios dentro del área geográfica de los datos reales.
    # Estos puntos enseñan al modelo a identificar zonas sin accidentes.
    # Duplicamos la cantidad de puntos positivos para un buen balance.
    n_ruido = int(len(df_pos) * 2.0) 
    lat_min, lat_max = df_reales['Start_Lat'].min(), df_reales['Start_Lat'].max()
    lng_min, lng_max = df_reales['Start_Lng'].min(), df_reales['Start_Lng'].max()
    
    # Generamos puntos aleatorios uniformes en el área
    df_neg_espacial = pd.DataFrame({
        'Start_Lat': np.random.uniform(lat_min, lat_max, n_ruido),
        'Start_Lng': np.random.uniform(lng_min, lng_max, n_ruido),
        'Hour': np.random.randint(0, 24, n_ruido),
        'DayOfWeek': np.random.randint(0, 7, n_ruido),
        'Weather_Code': np.random.choice(codigos_clima, n_ruido),
        'Probabilidad': 0.0
    })

    # Fusión y mezcla de todos los datos (Dataset, negativos temporales y espaciales)
    cols = ['Start_Lat', 'Start_Lng', 'Hour', 'DayOfWeek', 'Weather_Code', 'Probabilidad']
    df_final = pd.concat([df_pos[cols], df_neg_temporal[cols], df_neg_espacial[cols]], ignore_index=True)
    return df_final.sample(frac=1, random_state=42).reset_index(drop=True)

def entrenar_modelo(ciudad_sel):
    # Verifica si ya existe un modelo entrenado para la ciudad seleccionada
    global modelo_actual
    if modelo_actual["ciudad"] == ciudad_sel and modelo_actual["mlp"] is not None:
        return

    print(f"\n >>> ENTRENANDO NUEVO MODELO PARA: {ciudad_sel}")
    
    # Carga y preprocesamiento de datos
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"El archivo {DATASET_FILE} no existe. Se requiere dataset real.")

    # Leemos solo columnas clave para eficiencia + Description/Severity para el detalle
    df = pd.read_csv(DATASET_FILE, usecols=['Start_Lat', 'Start_Lng', 'Start_Time', 'Weather_Condition', 'City', 'Description', 'Severity'], nrows=600000)
    df = df[df['City'] == ciudad_sel].copy()
    
    if len(df) < 50: 
        raise ValueError(f"Pocos datos reales encontrados para {ciudad_sel}.")

    # Preprocesamiento básico y filtrado de columnas
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Hour'])
    
    # Encoding de Clima, texto a numérico
    le = LabelEncoder()
    climas_comunes = ['Clear', 'Rain', 'Cloudy', 'Fog', 'Snow', 'Overcast', 'Light Rain']
    all_weather = list(set(df['Weather_Condition'].unique()) | set(climas_comunes))
    le.fit(all_weather)
    df['Weather_Code'] = le.transform(df['Weather_Condition'])

    # Generacion de dataset probabilistico 
    df_entrenamiento = preparar_dataset_probabilistico(df)

    # Datos de entrada y salida
    X = df_entrenamiento[['Start_Lat', 'Start_Lng', 'Hour', 'DayOfWeek', 'Weather_Code']]
    y = df_entrenamiento['Probabilidad']

    # Escalado de caracteristicas, permite mejor convergencia del MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Parametros del MLP optimizados para este caso
    # Alpha 0.08: Regularizacion fuerte para suavizar la decisión y evitar sobreajuste.
    mlp = MLPRegressor(
        # hiden layers profundos para capturar complejidad espacial y temporal
        # Otra opcion es (100, 50, 25)
        # O más profundo pero más lento: (100, 80, 50, 25)
        hidden_layer_sizes=(100, 50, 25),
        activation='relu', 
        solver='adam', 
        alpha=0.08, 
        max_iter=500, 
        random_state=42
    )
    mlp.fit(X_scaled, y)

    # Construcción del KDTree para filtrado geografico rapido
    # Solo puntos con alta probabilidad de accidente
    # Esto ayuda a limitar las predicciones a calles y zonas transitadas.
    df_reales_high = df_entrenamiento[df_entrenamiento['Probabilidad'] >= 0.8]
    kdtree = KDTree(df_reales_high[['Start_Lat', 'Start_Lng']].values, leaf_size=40)

    # Guardar en estado global
    modelo_actual.update({
        "ciudad": ciudad_sel, 
        "mlp": mlp, 
        "scaler": scaler, 
        "encoder": le, 
        "kdtree": kdtree,
        "df": df # Guardamos el DF original para consultas de detalle
    })

# ==============================================================================
# LÓGICA DE VISUALIZACIÓN
# ==============================================================================

def obtener_color_riesgo(prob):
    if prob < 0.10: return '#2c3e50', 0.15 # Base Gris Oscuro (Latente)
    if prob < 0.20: return "#a0c9f1", 0.35 
    if prob < 0.30: return "#74d0ef", 0.40 
    if prob < 0.40: return "#46BABC", 0.45 
    if prob < 0.50: return '#2ecc71', 0.50 # Verde (Medio)
    if prob < 0.60: return '#f1c40f', 0.55 
    if prob < 0.70: return '#f39c12', 0.60 # Naranja (Alto)
    if prob < 0.80: return '#e67e22', 0.65 
    if prob < 0.90: return '#e74c3c', 0.75 # Rojo
    if prob < 0.95: return '#c0392b', 0.85 
    return '#8b0000', 0.95                 # Rojo Critico

def crear_mapa(ciudad, hora, dia, clima, check_dia):
    # Generar mapa
    entrenar_modelo(ciudad)
    
    # Configuracion inicial del mapa
    start_coords = CIUDADES.get(ciudad)
    m = folium.Map(location=start_coords, zoom_start=10, tiles='CartoDB dark_matter')

    # Configuracion de la Malla (Grid)
    center = CIUDADES.get(ciudad)
    resolucion = 180 # Alta resolución para "suavizado" visual
    
    # Limites de la malla (aprox 50km x 60km) + margen de 0.25/0.30 grados para cubrir periferia
    lat_min, lat_max = center[0]-0.25, center[0]+0.25
    lng_min, lng_max = center[1]-0.30, center[1]+0.30
    
    # Calculamos step y lo guardamos globalmente para el filtrado posterior
    lat_step = (lat_max - lat_min) / resolucion
    lng_step = (lng_max - lng_min) / resolucion
    modelo_actual["grid_steps"] = (lat_step, lng_step)

    # Generación de coordenadas de la malla
    # Crear una cuadrícula de puntos (lat, lng)
    lat_range = np.linspace(lat_min, lat_max, resolucion)
    lng_range = np.linspace(lng_min, lng_max, resolucion)
    grid_coords = np.array([[lat, lng] for lat in lat_range for lng in lng_range])
    
    # KDTREE Filtering Geográfico
    # AJUSTE: Reducir radio para evitar sobreestimación en zonas vacías
    RADIO_ACCIDENTE = 0.010 
    dist, _ = modelo_actual["kdtree"].query(grid_coords, k=1)
    mask_geo = dist.flatten() < RADIO_ACCIDENTE
    
    valid_points = grid_coords[mask_geo]
    dist_valid = dist.flatten()[mask_geo] # Distancias de los puntos válidos
    
    # Si no hay puntos válidos, retornamos el mapa base
    if len(valid_points) == 0: return m._repr_html_()

    # --- LÓGICA DE PREDICCIÓN  ---
    
    # Manejo de opciones clima
    w_codes = []
    if clima == 'Any':
        common_weather = ['Clear', 'Rain', 'Cloudy', 'Fog']
        # Obtener códigos para cada clima común
        for w in common_weather:
            try: w_codes.append(modelo_actual["encoder"].transform([w])[0])
            except: pass
        if not w_codes: w_codes = [0]
    else:
        # Clima específico, se obtiene su código
        try: w_codes = [modelo_actual["encoder"].transform([clima])[0]]
        except: w_codes = [0]

    # Manejo de horas para predicción
    # Si check_dia está activo, iteramos las 24 horas (range(24))
    horas_to_predict = range(24) if check_dia else [int(hora)]
    
    # Acumuladores para cálculo ponderado
    max_probs = np.zeros(len(valid_points))
    mean_probs_accum = np.zeros(len(valid_points))
    total_iterations = 0

    # Bucle de inferencia múltiple 
    for h in horas_to_predict:
        for w in w_codes:
            # Construir vector de entrada para el MLP
            X_q = np.zeros((len(valid_points), 5))
            X_q[:, 0] = valid_points[:, 0]
            X_q[:, 1] = valid_points[:, 1]
            X_q[:, 2] = h
            X_q[:, 3] = int(dia)
            X_q[:, 4] = w
            
            # ARREGLO WARNING: Convertir a DataFrame con nombres antes de transformar
            X_q_df = pd.DataFrame(X_q, columns=['Start_Lat', 'Start_Lng', 'Hour', 'DayOfWeek', 'Weather_Code'])
            
            # Escalado y prediccionb de probabilidades
            X_q_scaled = modelo_actual["scaler"].transform(X_q_df)
            probs = modelo_actual["mlp"].predict(X_q_scaled)
            
            # Guardar maximos y acumulados para ponderación final
            max_probs = np.maximum(max_probs, probs)
            mean_probs_accum += probs
            total_iterations += 1
    
    # Calculo de probabilidades finales ponderadas
    # Se da mas peso al promedio para suavizar picos aislados, pero se considera el maximo para no perder riesgos puntuales
    avg_probs = mean_probs_accum / total_iterations
    prob_preds = (avg_probs * 0.6) + (max_probs * 0.4)
    
    # AJUSTE SOBREESTIMACIÓN: Penalización por distancia
    # Si estás en el borde del radio (lejos del accidente real), bajamos la prob
    decay_factor = 1.0 - (dist_valid / RADIO_ACCIDENTE) # 1 si estas encima, 0 si estas al limite
    prob_preds = prob_preds * np.clip(decay_factor, 0.2, 1.0) # Nunca bajar de 0.2 el factor

    # ARREGLO ERROR MATEMÁTICO: Clip para evitar negativos antes del power
    prob_preds = np.clip(prob_preds, 0, 1)
    
    # Suavizado exponencial para limpiar ruido de fondo
    # .power hace que valores bajos se reduzcan y altos se mantengan o aumenten
    prob_preds = np.power(prob_preds, 1.25)
    
    # --- CONSTRUCCIÓN VISUAL ---
    
    # Capa de riesgo, visualización vectorial
    riesgo_layer = FeatureGroup(name="Capa_Riesgo", overlay=True)
    
    for i, prob in enumerate(prob_preds):
        val = max(0.0, min(1.0, prob))
        
        # Filtro de limpieza visual (suelo mínimo)
        if val < 0.1: val = 0.05
            
        # Obtener color y opacidad según nivel de riesgo
        color, opacity = obtener_color_riesgo(val)
        lat, lng = valid_points[i][0], valid_points[i][1]
        
        # Dibujar rectángulo vectorial
        # Se añade una clase CSS personalizada (aunque Folium no lo soporta nativamente fácil, 
        # el script inyectado interceptará el evento click sobre los objetos Leaflet).
        rect = folium.Rectangle(
            bounds=[[lat, lng], [lat + lat_step, lng + lng_step]],
            color=None, # Sin borde para rendimiento
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=f"{val*100:.0f}%" if val > 0.15 else None 
        )
        rect.add_to(riesgo_layer)

    riesgo_layer.add_to(m)
    
    # Inyección de JS para interactividad (Toggle + Centrar + Click en Grilla)
    lat_c, lng_c = start_coords
    m.get_root().html.add_child(folium.Element(f"""
        <script>
            var riskLayerRef = null; 
            
            // Función auxiliar para notificar al padre
            function notifyParent(lat, lng) {{
                if (parent && parent.loadAccidents) {{
                    parent.loadAccidents(lat, lng);
                }}
            }}

            // Función para toggle capa
            window.toggleLayerGlobal = function() {{
                var mapInstance;
                for (var key in window) {{
                    if (key.startsWith('map_')) {{
                        mapInstance = window[key];
                        break;
                    }}
                }}
                if (!mapInstance) return;

                if (!riskLayerRef) {{
                    mapInstance.eachLayer(function(layer) {{
                        if (layer instanceof L.FeatureGroup && layer !== mapInstance) {{
                            riskLayerRef = layer;
                            // Attach click event to all layers inside this group
                            riskLayerRef.eachLayer(function(l) {{
                                l.on('click', function(e) {{
                                    notifyParent(l.getBounds().getSouthWest().lat, l.getBounds().getSouthWest().lng);
                                }});
                            }});
                        }}
                    }});
                }}

                if (riskLayerRef) {{
                    if (mapInstance.hasLayer(riskLayerRef)) {{
                        mapInstance.removeLayer(riskLayerRef);
                    }} else {{
                        mapInstance.addLayer(riskLayerRef);
                    }}
                }}
            }};

            // Función para centrar mapa
            window.centerMapGlobal = function() {{
                var mapInstance;
                for (var key in window) {{
                    if (key.startsWith('map_')) {{
                        mapInstance = window[key];
                        break;
                    }}
                }}
                if (mapInstance) {{
                    mapInstance.setView([{lat_c}, {lng_c}], 10);
                }}
            }};
            
            // Inicializar clicks al cargar
            document.addEventListener("DOMContentLoaded", function() {{
                setTimeout(function(){{
                    var mapInstance;
                    for (var key in window) {{
                        if (key.startsWith('map_')) {{
                            mapInstance = window[key];
                            break;
                        }}
                    }}
                    if(mapInstance) {{
                         mapInstance.eachLayer(function(layer) {{
                            if (layer instanceof L.FeatureGroup && layer !== mapInstance) {{
                                layer.eachLayer(function(l) {{
                                    l.on('click', function(e) {{
                                        notifyParent(l.getBounds().getSouthWest().lat, l.getBounds().getSouthWest().lng);
                                    }});
                                }});
                            }}
                        }});
                    }}
                }}, 1000);
            }});
        </script>
    """))

    # Retornar mapa como HTML embebido
    return m._repr_html_()

# Rutas de Flask
@app.route("/", methods=["GET", "POST"])
def index():
    # Parámetros por defecto
    params = {
        'ciudad': 'Los Angeles', 
        'hora': 18, 
        'dia': 4, 
        'clima': 'Clear',
        'check_dia': False
    }
    
    if request.method == "POST":
        params['ciudad'] = request.form.get("ciudad")
        params['hora'] = int(request.form.get("hora"))
        params['dia'] = int(request.form.get("dia"))
        params['clima'] = request.form.get("clima")
        params['check_dia'] = request.form.get("check_dia") == 'on'
    
    mapa = crear_mapa(params['ciudad'], params['hora'], params['dia'], params['clima'], params['check_dia'])
    
    return render_template('index.html', mapa_html=mapa, ciudades_list=CIUDADES.keys(), 
                                  ciudad_sel=params['ciudad'], hora=params['hora'], 
                                  dia=params['dia'], clima=params['clima'], check_dia=params['check_dia'])

@app.route("/get_accidents", methods=["POST"])
def get_accidents():
    # Endpoint AJAX para obtener lista de accidentes
    data = request.get_json()
    lat_req = float(data.get('lat'))
    lng_req = float(data.get('lng'))
    
    if modelo_actual["df"] is None or modelo_actual["grid_steps"] is None:
        return jsonify({"error": "Modelo no cargado"})
        
    df = modelo_actual["df"]
    lat_step, lng_step = modelo_actual["grid_steps"]
    
    # Filtrar dataframe dentro del bounding box del cuadrante clickeado
    # Se usa un pequeño epsilon para asegurar bordes
    mask = (
        (df['Start_Lat'] >= lat_req - 0.0001) & 
        (df['Start_Lat'] < lat_req + lat_step + 0.0001) &
        (df['Start_Lng'] >= lng_req - 0.0001) & 
        (df['Start_Lng'] < lng_req + lng_step + 0.0001)
    )
    
    subset = df[mask].head(50) # Limitamos a 50 para no saturar UI
    
    # Diccionario simple de traducción
    WEATHER_TRANSLATION = {
        'Clear': 'Despejado', 'Fair': 'Despejado', 'Cloudy': 'Nublado', 'Mostly Cloudy': 'Mayormente Nublado',
        'Partly Cloudy': 'Parcialmente Nublado', 'Rain': 'Lluvia', 'Light Rain': 'Lluvia Ligera',
        'Heavy Rain': 'Lluvia Fuerte', 'Snow': 'Nieve', 'Light Snow': 'Nieve Ligera',
        'Fog': 'Niebla', 'Haze': 'Neblina', 'Thunderstorm': 'Tormenta', 'Overcast': 'Cubierto',
        'Mist': 'Neblina', 'Scattered Clouds': 'Nubes Dispersas'
    }

    resultado = []
    for _, row in subset.iterrows():
        # Procesamos fecha y hora por separado, ignoramos severidad
        dt = pd.to_datetime(row['Start_Time'])
        weather_en = row['Weather_Condition']
        weather_es = WEATHER_TRANSLATION.get(weather_en, weather_en) # Fallback al inglés si no encuentra
        
        resultado.append({
            "date": dt.strftime('%Y-%m-%d'),
            "time": dt.strftime('%H:%M'),
            "weather": weather_es,
            "desc": row.get('Description', 'Sin descripción')
        })
        
    return jsonify(resultado)

@app.route("/get_dashboard_stats", methods=["GET"])
def get_dashboard_stats():
    # Calcular estadisticas globales para la ciudad cargada
    if modelo_actual["df"] is None:
        return jsonify({"error": "Datos no cargados"})
    
    df = modelo_actual["df"]
    
    # Total Accidentes
    total = len(df)
    
    # Crecimiento anual
    df['Year'] = df['Start_Time'].dt.year
    counts_by_year = df['Year'].value_counts().sort_index()
    
    growth_rate = 0
    if len(counts_by_year) >= 2:
        last_year_count = counts_by_year.iloc[-1]
        prev_year_count = counts_by_year.iloc[-2]
        if prev_year_count > 0:
            growth_rate = ((last_year_count - prev_year_count) / prev_year_count) * 100
            
    # Top Climas
    top_weather_raw = df['Weather_Condition'].value_counts().head(4)
    
    # Reutilizamos traduccion
    WEATHER_TRANSLATION = {
        'Clear': 'Despejado', 'Fair': 'Despejado', 'Cloudy': 'Nublado', 'Mostly Cloudy': 'Mayormente Nublado',
        'Partly Cloudy': 'Parcialmente Nublado', 'Rain': 'Lluvia', 'Light Rain': 'Lluvia Ligera',
        'Heavy Rain': 'Lluvia Fuerte', 'Snow': 'Nieve', 'Light Snow': 'Nieve Ligera',
        'Fog': 'Niebla', 'Haze': 'Neblina', 'Thunderstorm': 'Tormenta', 'Overcast': 'Cubierto',
        'Mist': 'Neblina', 'Scattered Clouds': 'Nubes Dispersas'
    }
    
    top_weather = []
    for w, c in top_weather_raw.items():
        top_weather.append([WEATHER_TRANSLATION.get(w, w), int(c)]) # int para serializacion json
        
    # Dia Mas Peligroso
    days_map = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
    day_idx = df['DayOfWeek'].value_counts().idxmax()
    most_common_day = days_map.get(day_idx, "N/A")
    
    # Distribución Día/Noche (6am - 6pm)
    day_accidents = len(df[(df['Hour'] >= 6) & (df['Hour'] < 18)])
    day_pct = round((day_accidents / total) * 100, 1)

    return jsonify({
        "total": int(total),
        "growth": growth_rate,
        "top_weather": top_weather,
        "most_common_day": most_common_day,
        "day_pct": day_pct
    })

if __name__ == "__main__":
    app.run(debug=False, port=8080)
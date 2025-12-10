# ==============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import os
import time
import pandas as pd
import numpy as np
# Flask y folium para la app web
from flask import Flask, render_template_string, request
import folium
from folium import FeatureGroup
# Sklearn provee modelos y utilidades ML
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KDTree

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
    "kdtree": None
}

# ==============================================================================
# LÓGICA DE MACHINE LEARNING
# ==============================================================================

def generar_datos_sinteticos_fallback(ciudad):
    # Genera un dataset sintético simple si no hay datos reales.
    # En caso de no encontrar datos, usamos una distribución uniforme
    print(f" [ALERTA] Usando modo demostración (Sintético) para {ciudad}...")
    center = CIUDADES.get(ciudad, [34.05, -118.24])
    n_samples = 4000
    # Aumentamos dispersión para cubrir más área en modo demo
    lat_base = np.random.uniform(center[0]-0.25, center[0]+0.25, n_samples)
    lng_base = np.random.uniform(center[1]-0.3, center[1]+0.3, n_samples)
    
    data = {
        'Start_Lat': lat_base,
        'Start_Lng': lng_base,
        'Start_Time': pd.date_range(start='1/1/2023', periods=n_samples, freq='h'),
        'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Cloudy'], n_samples)
    }
    return pd.DataFrame(data)

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
    if os.path.exists(DATASET_FILE):
        try:
            # Leemos solo columnas clave para eficiencia
            df = pd.read_csv(DATASET_FILE, usecols=['Start_Lat', 'Start_Lng', 'Start_Time', 'Weather_Condition', 'City'], nrows=600000)
            df = df[df['City'] == ciudad_sel].copy()
            if len(df) < 50: raise ValueError("Pocos datos")
        except:
            df = generar_datos_sinteticos_fallback(ciudad_sel)
    else:
        df = generar_datos_sinteticos_fallback(ciudad_sel)

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
    df_reales = df_entrenamiento[df_entrenamiento['Probabilidad'] >= 0.8]
    kdtree = KDTree(df_reales[['Start_Lat', 'Start_Lng']].values, leaf_size=40)

    # Guardar en estado global
    modelo_actual.update({"ciudad": ciudad_sel, "mlp": mlp, "scaler": scaler, "encoder": le, "kdtree": kdtree})

# ==============================================================================
# LÓGICA DE VISUALIZACIÓN
# ==============================================================================

def obtener_color_riesgo(prob):
    if prob < 0.10: return '#2c3e50', 0.15 # Base Gris Oscuro (Latente)
    if prob < 0.20: return '#1e90ff', 0.35 
    if prob < 0.30: return '#00bfff', 0.40 
    if prob < 0.40: return '#00ced1', 0.45 
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
    
    # Generación de coordenadas de la malla
    # Crear una cuadrícula de puntos (lat, lng)
    lat_range = np.linspace(lat_min, lat_max, resolucion)
    lng_range = np.linspace(lng_min, lng_max, resolucion)
    grid_coords = np.array([[lat, lng] for lat in lat_range for lng in lng_range])
    
    # KDTREE Filtering Geográfico
    # Solo predecimos para puntos cercanos a historial real (calles).
    # Esto elimina artefactos en el mar o desierto.
    dist, _ = modelo_actual["kdtree"].query(grid_coords, k=1)
    mask_geo = dist.flatten() < 0.012 
    valid_points = grid_coords[mask_geo]
    
    # Si no hay puntos válidos, retornamos el mapa base
    if len(valid_points) == 0: return m._repr_html_()

    # --- LÓGICA DE PREDICCIÓN AVANZADA ---
    
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
            
            # Escalado y prediccionb de probabilidades
            X_q_scaled = modelo_actual["scaler"].transform(X_q)
            probs = modelo_actual["mlp"].predict(X_q_scaled)
            
            # Guardar maximos y acumulados para ponderación final
            max_probs = np.maximum(max_probs, probs)
            mean_probs_accum += probs
            total_iterations += 1
    
    # Calculo de probabilidades finales ponderadas
    # Se da mas peso al promedio para suavizar picos aislados, pero se considera el maximo para no perder riesgos puntuales
    avg_probs = mean_probs_accum / total_iterations
    prob_preds = (avg_probs * 0.6) + (max_probs * 0.4)
    
    # Suavizado exponencial para limpiar ruido de fondo
    # .power hace que valores bajos se reduzcan y altos se mantengan o aumenten
    prob_preds = np.power(prob_preds, 1.25)
    
    # --- CONSTRUCCIÓN VISUAL ---
    
    # Capa de riesgo, visualización vectorial
    riesgo_layer = FeatureGroup(name="Capa_Riesgo", overlay=True)
    
    # Pre-calculo de pasos para dibujar rectángulos
    lat_step = (lat_max - lat_min) / resolucion
    lng_step = (lng_max - lng_min) / resolucion

    for i, prob in enumerate(prob_preds):
        val = max(0.0, min(1.0, prob))
        
        # Filtro de limpieza visual (suelo mínimo)
        if val < 0.1: val = 0.05
            
        # Obtener color y opacidad según nivel de riesgo
        color, opacity = obtener_color_riesgo(val)
        lat, lng = valid_points[i][0], valid_points[i][1]
        
        # Dibujar rectángulo vectorial
        folium.Rectangle(
            bounds=[[lat, lng], [lat + lat_step, lng + lng_step]],
            color=None, # Sin borde para rendimiento
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=f"{val*100:.0f}%" if val > 0.15 else None 
        ).add_to(riesgo_layer)

    riesgo_layer.add_to(m)
    
    # Inyección de JS para interactividad (Toggle + Centrar)
    lat_c, lng_c = start_coords
    m.get_root().html.add_child(folium.Element(f"""
        <script>
            var riskLayerRef = null; 
            // Función para toggle capa
            window.toggleLayerGlobal = function() {{
                var mapInstance;
                for (var key in window) {{
                    if (key.startsWith('map_')) {{
                        // Encontrar instancia del mapa
                        mapInstance = window[key];
                        break;
                    }}
                }}
                if (!mapInstance) return;

                if (!riskLayerRef) {{
                    mapInstance.eachLayer(function(layer) {{
                        // Buscar la capa de riesgo por tipo
                        if (layer instanceof L.FeatureGroup && layer !== mapInstance) {{
                            riskLayerRef = layer;
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
        </script>
    """))

    # Retornar mapa como HTML embebido
    return m._repr_html_()


# ==============================================================================
# FRONTEND (HTML EMBEBIDO)
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Predicción de Accidentes</title>
    <!-- Bootstrap 5 & Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css">
    <style>
        body, html { height: 100%; margin: 0; overflow: hidden; background: #121212; color: #e0e0e0; }
        .sidebar {
            height: 100vh;
            background-color: #1e1e1e;
            padding: 25px;
            border-right: 1px solid #333;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        .map-wrapper { height: 100vh; padding: 0; position: relative; }
        iframe { width: 100%; height: 100%; border: none; display: block; }
        
        .btn-predict {
            width: 100%;
            background: linear-gradient(135deg, #0d6efd, #0dcaf0);
            border: none;
            padding: 12px;
            font-weight: 600;
            color: white;
            margin-top: 20px;
            border-radius: 8px;
        }
        .btn-predict:hover { opacity: 0.9; transform: translateY(-1px); }
        
        .btn-toggle {
            background-color: #2c2c2c;
            color: #b0b0b0;
            border: 1px solid #444;
            padding: 10px;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .btn-toggle:hover { background-color: #383838; color: white; border-color: #666; }

        .btn-center {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 9999;
            background: #1e1e1e;
            color: white;
            border: 1px solid #444;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            cursor: pointer;
        }
        .btn-center:hover { background: #333; }

        #loadingOverlay {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            z-index: 2000;
            display: none;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .spinner-border { width: 3rem; height: 3rem; }

        label { color: #aaa; margin-bottom: 5px; font-size: 0.85rem; font-weight: 500; }
        .form-select, .form-range { background-color: #2c2c2c; color: white; border: 1px solid #444; }
        .form-select:focus { background-color: #2c2c2c; color: white; border-color: #0d6efd; box-shadow: none; }
        
        .color-box { display: inline-block; width: 10px; height: 10px; margin-right: 8px; border-radius: 50%; }
        .legend-item { font-size: 0.75rem; color: #bbb; display: flex; align-items: center; margin-bottom: 3px; }
        
        h3 { font-weight: 700; letter-spacing: -0.5px; }
    </style>
</head>
<body>

<div id="loadingOverlay">
    <div class="spinner-border text-info mb-3" role="status"></div>
    <h4 class="text-white">Procesando Datos...</h4>
    <small class="text-muted">Analizando Dataset y cargango MLP</small>
</div>

<div class="container-fluid">
    <div class="row">
        <!-- BARRA LATERAL -->
        <div class="col-md-3 sidebar">
            <div class="text-center mb-4">
                <h3 class="mb-1"><i class="bi bi-cpu"></i> MLP</h3>
                <span class="badge bg-secondary opacity-50">Sistema de prediccón de accidentes USA</span>
            </div>
            
            <form action="/" method="post" id="mainForm" onsubmit="showLoading()">
                <div class="mb-3">
                    <label>Ciudad de estudio</label>
                    <select class="form-select" name="ciudad" onchange="showLoading(); this.form.submit()">
                        {% for c in ciudades_list %}
                        <option value="{{ c }}" {% if c == ciudad_sel %}selected{% endif %}>{{ c }}</option>
                        {% endfor %}
                    </select>
                </div>

                <!-- Checkbox Solo Día -->
                <div class="form-check form-switch mb-3">
                    <input class="form-check-input" type="checkbox" id="checkDia" name="check_dia" 
                           {% if check_dia %}checked{% endif %} onchange="toggleHourInput()">
                    <label class="form-check-label" for="checkDia">Predicción general por día</label>
                </div>

                <div class="mb-3" id="hourInputGroup">
                    <label>Hora: <span class="text-info" id="hourDisplay">{{ hora }}:00</span></label>
                    <input type="range" class="form-range" min="0" max="23" name="hora" value="{{ hora }}" 
                           oninput="document.getElementById('hourDisplay').innerText = this.value + ':00'">
                </div>

                <div class="mb-3">
                    <label>Día de la semana</label>
                    <select class="form-select" name="dia">
                        <option value="0" {% if dia==0 %}selected{% endif %}>Lunes</option>
                        <option value="1" {% if dia==1 %}selected{% endif %}>Martes</option>
                        <option value="2" {% if dia==2 %}selected{% endif %}>Miércoles</option>
                        <option value="3" {% if dia==3 %}selected{% endif %}>Jueves</option>
                        <option value="4" {% if dia==4 %}selected{% endif %}>Viernes</option>
                        <option value="5" {% if dia==5 %}selected{% endif %}>Sábado</option>
                        <option value="6" {% if dia==6 %}selected{% endif %}>Domingo</option>
                    </select>
                </div>

                <div class="mb-3">
                    <label>Condición climática</label>
                    <select class="form-select" name="clima">
                        <option value="Any" {% if clima=='Any' %}selected{% endif %}>🌍 Cualquiera (Promedio)</option>
                        <option value="Clear" {% if clima=='Clear' %}selected{% endif %}>☀️ Despejado</option>
                        <option value="Rain" {% if clima=='Rain' %}selected{% endif %}>🌧️ Lluvia</option>
                        <option value="Cloudy" {% if clima=='Cloudy' %}selected{% endif %}>🌥️ Nublado</option>
                        <option value="Fog" {% if clima=='Fog' %}selected{% endif %}>🌫️ Niebla</option>
                        <option value="Snow" {% if clima=='Snow' %}selected{% endif %}>❄️ Nieve</option>
                    </select>
                </div>

                <button type="submit" class="btn btn-predict shadow-sm">
                    <i class="bi bi-lightning-charge"></i> Calcular Riesgo
                </button>
            </form>

            <hr class="border-secondary my-4">

            <button class="btn btn-toggle w-100 mb-3" onclick="toggleMapLayer()">
                <i class="bi bi-eye"></i> <span id="btnText">Ocultar mapa</span>
            </button>

            <!-- Leyenda Detallada -->
            <div class="p-3 rounded" style="background: rgba(255,255,255,0.05);">
                <label class="mb-2 text-uppercase small" style="letter-spacing:1px;">Probabilidad</label>
                <div class="legend-item"><span class="color-box" style="background:#2c3e50;"></span> < 10% (Base)</div>
                <div class="legend-item"><span class="color-box" style="background:#1e90ff;"></span> 10-20%</div>
                <div class="legend-item"><span class="color-box" style="background:#00bfff;"></span> 20-30%</div>
                <div class="legend-item"><span class="color-box" style="background:#00ced1;"></span> 30-40%</div>
                <div class="legend-item"><span class="color-box" style="background:#2ecc71;"></span> 40-50%</div>
                <div class="legend-item"><span class="color-box" style="background:#f1c40f;"></span> 50-60%</div>
                <div class="legend-item"><span class="color-box" style="background:#f39c12;"></span> 60-70%</div>
                <div class="legend-item"><span class="color-box" style="background:#e67e22;"></span> 70-80%</div>
                <div class="legend-item"><span class="color-box" style="background:#e74c3c;"></span> 80-90%</div>
                <div class="legend-item"><span class="color-box" style="background:#c0392b;"></span> 90-95%</div>
                <div class="legend-item"><span class="color-box" style="background:#8b0000;"></span> > 95% (Crítico)</div>
            </div>
        </div>
        
        <!-- VISUALIZACIÓN -->
        <div class="col-md-9 map-wrapper">
            <button class="btn-center" onclick="centerMap()" title="Centrar Mapa">
                <i class="bi bi-crosshair"></i>
            </button>
            {{ mapa_html|safe }}    
        </div>
    </div>
</div>

<script>
    function toggleHourInput() {
        var checkbox = document.getElementById('checkDia');
        var group = document.getElementById('hourInputGroup');
        if (checkbox.checked) {
            group.style.opacity = '0.3';
            group.style.pointerEvents = 'none';
        } else {
            group.style.opacity = '1';
            group.style.pointerEvents = 'auto';
        }
    }
    toggleHourInput();

    function showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    function toggleMapLayer() {
        var iframe = document.querySelector('iframe');
        if (iframe && iframe.contentWindow && typeof iframe.contentWindow.toggleLayerGlobal === "function") {
            iframe.contentWindow.toggleLayerGlobal();
            var btnText = document.getElementById("btnText");
            if (btnText.innerText.includes("Ocultar")) {
                btnText.innerText = "Mostrar Capa";
            } else {
                btnText.innerText = "Ocultar Capa";
            }
        }
    }

    function centerMap() {
        var iframe = document.querySelector('iframe');
        if (iframe && iframe.contentWindow && typeof iframe.contentWindow.centerMapGlobal === "function") {
            iframe.contentWindow.centerMapGlobal();
        }
    }
</script>
</body>
</html>
"""

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
    
    return render_template_string(HTML_TEMPLATE, mapa_html=mapa, ciudades_list=CIUDADES.keys(), 
                                  ciudad_sel=params['ciudad'], hora=params['hora'], 
                                  dia=params['dia'], clima=params['clima'], check_dia=params['check_dia'])

if __name__ == "__main__":
    app.run(debug=False, port=5000)
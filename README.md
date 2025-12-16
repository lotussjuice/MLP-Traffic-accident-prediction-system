# Sistema de Predicción de Accidentes de Tráfico (MLP)

Este proyecto corresponde a una **aplicación web interactiva** orientada a la **estimación y visualización del riesgo de accidentes de tráfico** en distintas ciudades de Estados Unidos. Su objetivo principal es apoyar el análisis preventivo mediante el uso de técnicas de **Inteligencia Artificial**, facilitando la comprensión espacial y temporal de los accidentes automovilísticos.

El sistema utiliza una **Red Neuronal Perceptrón Multicapa (MLP)** entrenada con millones de registros históricos de accidentes, lo que permite estimar la probabilidad de ocurrencia de siniestros considerando múltiples variables relevantes, entre ellas:

* Ubicación geográfica
* Hora del día
* Día de la semana
* Condiciones climáticas

La combinación de estas variables permite generar predicciones dinámicas y ajustables en tiempo real a través de una interfaz web intuitiva.

---

## Características Principales

### Mapa de Calor de Riesgo

El sistema presenta un **mapa de calor interactivo** que permite identificar visualmente las zonas con mayor o menor riesgo de accidentes. Los niveles de riesgo se representan mediante una escala de colores progresiva, donde los tonos fríos indican bajo riesgo y los tonos cálidos representan áreas críticas. Esta visualización facilita la detección rápida de patrones espaciales de peligrosidad vial.

### Predicciones Personalizadas

El usuario puede modificar distintos parámetros para observar cómo varía el riesgo estimado de accidentes, entre ellos:

* Hora exacta del día
* Día de la semana
* Condiciones climáticas como lluvia, nieve, niebla u otras

Cada cambio en estos parámetros provoca una actualización inmediata de las predicciones, permitiendo analizar distintos escenarios de forma comparativa.

### Consulta Histórica de Accidentes

Además de las predicciones, el sistema permite consultar información histórica real. Al hacer clic sobre un sector específico del mapa, se despliega una lista detallada de los accidentes que han ocurrido en esa zona, incluyendo información como la fecha, la hora y la localización aproximada. Esto permite contrastar las predicciones del modelo con eventos reales registrados.

### Dashboard Estadístico

El sistema incluye un panel estadístico que resume información relevante para la ciudad seleccionada, tales como:

* Número total de accidentes registrados
* Tasa de crecimiento anual de accidentes
* Condiciones climáticas asociadas a mayor peligrosidad
* Distribución de accidentes según horario diurno y nocturno

Este dashboard proporciona una visión global que complementa el análisis geográfico del mapa.

### Filtrado por Ciudad

Actualmente, el sistema ofrece soporte para las siguientes ciudades de Estados Unidos:

* Los Angeles
* New York
* Chicago
* Miami
* Houston

Cada ciudad cuenta con su propio conjunto de datos y un modelo ajustado a sus características particulares.

---

## Requisitos Previos

Antes de ejecutar el sistema, es necesario contar con los siguientes requisitos:

* Python versión 3.8 o superior
* Acceso a una terminal o consola de comandos
* Instalación de las librerías necesarias, las cuales se detallan más adelante

---

## Instalación y Configuración del Dataset

Este paso es fundamental para el correcto funcionamiento del sistema. Debido al gran volumen de información histórica utilizada, el dataset no se incluye directamente en el repositorio del proyecto y debe ser descargado manualmente.

### Descarga del Dataset

El sistema utiliza el dataset **"US Accidents"**, disponible públicamente en la plataforma Kaggle.

Pasos a seguir:

1. Acceder al enlace correspondiente al dataset **US Accidents (2016 – 2023)**.
2. Descargar el archivo comprimido (ZIP).
3. Descomprimir el archivo descargado.
4. Identificar el archivo denominado:

```text
US_Accidents_March23.csv
```

5. Mover dicho archivo a la carpeta raíz del proyecto, asegurándose de que se encuentre al mismo nivel que el archivo `app.py`.

---

### Estructura de Archivos

Para que la aplicación funcione correctamente, la estructura del proyecto debe respetar el siguiente formato:

```text
/TU_CARPETA_PROYECTO
  ├── app.py                    # Código principal del backend
  ├── US_Accidents_March23.csv  # Dataset requerido por el sistema
  ├── templates/
  │     └── index.html          # Interfaz de usuario
  └── static/
        ├── css/
        │     └── styles.css    # Estilos visuales
        └── js/
              └── main.js       # Lógica del frontend
```

---

### Instalación de Dependencias

Con la terminal ubicada en la carpeta del proyecto, ejecutar el siguiente comando para instalar las dependencias necesarias:

```bash
pip install pandas numpy flask folium scikit-learn
```

Estas librerías permiten el procesamiento de datos, el entrenamiento del modelo de aprendizaje automático y la visualización interactiva.

---

## Ejecución del Sistema

Una vez completada la instalación y configuración del dataset, se puede iniciar la aplicación ejecutando:

```bash
python app.py
```

Al iniciar, el sistema mostrará un mensaje indicando que el servidor web se encuentra en ejecución. La carga inicial puede tardar algunos segundos, ya que se procesan y preparan los datos necesarios.

Posteriormente, se debe abrir un navegador web y acceder a la siguiente dirección:

```text
http://127.0.0.1:8080
```

---

## Uso del Sistema

1. **Selección de ciudad**: desde la barra lateral izquierda, el usuario elige la ciudad que desea analizar.
2. **Configuración de parámetros**: se ajustan la hora, el día de la semana y las condiciones climáticas.
3. **Cálculo del riesgo**: al presionar el botón correspondiente, el sistema actualiza el mapa con las nuevas predicciones.
4. **Exploración del mapa**: los colores indican el nivel de riesgo estimado; al hacer clic sobre una zona se puede consultar el historial de accidentes reales.
5. **Visualización de estadísticas**: mediante el botón "Ver Estadísticas" se accede a un resumen general de la ciudad seleccionada.

---

## Consideraciones sobre el Rendimiento

La primera vez que se ejecuta el sistema o se selecciona una ciudad, el modelo puede tardar algunos segundos en entrenarse. Este comportamiento es normal, ya que el sistema procesa una gran cantidad de datos históricos para generar predicciones confiables. Una vez finalizado este proceso inicial, la interacción con la aplicación es fluida y responsiva.

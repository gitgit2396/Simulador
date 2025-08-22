# Simulador Económico Planta Fotovoltaica

Aplicación web desarrollada en Python con Streamlit para simular la economía de una planta fotovoltaica con horizonte de 30 años, incluyendo estructura de deuda, depreciaciones, repotenciación y análisis multi-escenario.

## Características Principales

- **Tres modos de ejecución:**
  - Cargar parámetros desde archivo YAML
  - Modificar parámetros manualmente mediante formulario
  - Ejecutar múltiples escenarios simultáneamente

- **Análisis financiero completo:**
  - Cálculo de flujos de caja libres (FCFF) y al equity (FCFE)
  - KPIs: NPV, IRR, Payback para proyecto y equity
  - Estructura de deuda con cuota francesa
  - Análisis de cobertura de servicio de deuda (DSCR)

- **Funcionalidades avanzadas:**
  - Análisis de sensibilidad tipo tornado
  - Dimensionamiento óptimo de deuda por DSCR objetivo
  - Comparador multi-escenario
  - Descargas opcionales en CSV/XLSX (sin escribir al disco)

- **Interfaz intuitiva:**
  - Pestañas organizadas: KPIs, Data, Calendario Deuda, Inputs, Gráficos
  - Visualizaciones interactivas con Plotly
  - Validación suave de parámetros con warnings
  - Formateo automático de números

## Instalación y Ejecución

### Requisitos
- Python 3.11 o superior
- pip (gestor de paquetes de Python)

### Pasos para ejecutar

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación:**
   ```bash
   streamlit run app8_web.py
   ```

3. **Abrir en el navegador:**
   La aplicación se abrirá automáticamente en `http://localhost:8501`

## Estructura del Proyecto

```
├── app8_web.py          # Aplicación principal Streamlit
├── model_core.py        # Módulo con lógica financiera
├── requirements.txt     # Dependencias del proyecto
└── README.md           # Este archivo
```

## Uso de la Aplicación

### Modo 1: Ejecutar con datos cargados (YAML)
1. Seleccionar "Ejecutar con datos cargados (YAML)" en el sidebar
2. Subir un archivo YAML con los parámetros del proyecto
3. La aplicación ejecutará automáticamente el modelo y mostrará los resultados

### Modo 2: Ejecutar modificando parámetros
1. Seleccionar "Ejecutar modificando parámetros" en el sidebar
2. Opcionalmente cargar un YAML base para precargar valores
3. Modificar los parámetros en el formulario interactivo
4. Hacer clic en "Ejecutar Escenario"
5. Opcionalmente descargar la configuración como YAML

### Modo 3: Ejecutar todos los escenarios a la vez
1. Seleccionar "Ejecutar todos los escenarios a la vez" en el sidebar
2. Subir múltiples archivos YAML
3. Hacer clic en "Ejecutar Todos los Escenarios"
4. Usar el comparador multi-escenario para analizar resultados

## Parámetros del Modelo

### Técnicos
- **YEARS**: Horizonte del proyecto (20-35 años)
- **PLANT_MW**: Capacidad instalada (1-300 MW)
- **CAP_FACTOR**: Factor de capacidad (0.18-0.33)
- **DEGRADATION**: Degradación anual (0.003-0.008)

### Económicos
- **CAPEX0_PER_KW**: CapEx inicial (500-1200 USD/kW)
- **PRICE_USD_PER_MWH_Y1**: Precio energía año 1 (25-70 USD/MWh)
- **PRICE_ESCALATION**: Escalación precio (0-3% anual)

### Operativos
- **FOM_USD_PER_KW_YR**: O&M fijo (8-25 USD/kW/año)
- **VOM_USD_PER_MWH**: O&M variable (0-3 USD/MWh)
- **OPEX_ESCALATION**: Escalación OpEx (0-3% anual)

### Financieros
- **DEBT_PCT_CAPEX0**: Nivel de deuda (0-60% del CapEx)
- **DEBT_RATE**: Tasa de interés (6-14% anual)
- **DEBT_TERM_YEARS**: Plazo de la deuda (8-18 años)
- **WACC**: Costo promedio ponderado de capital (7-13%)
- **KE**: Costo del equity (12-20%)
- **TAX_RATE**: Tasa de impuestos (25-35%)

## Ejemplo de Archivo YAML

```yaml
# Parámetros técnicos
YEARS: 30
PLANT_MW: 100.0
CAP_FACTOR: 0.28
DEGRADATION: 0.005

# Parámetros económicos
CAPEX0_PER_KW: 750.0
PRICE_USD_PER_MWH_Y1: 50.0
PRICE_ESCALATION: 0.02

# Parámetros operativos
FOM_USD_PER_KW_YR: 12.0
VOM_USD_PER_MWH: 2.0
OPEX_ESCALATION: 0.025

# Parámetros financieros
DEBT_PCT_CAPEX0: 0.40
DEBT_RATE: 0.075
DEBT_TERM_YEARS: 15
WACC: 0.095
KE: 0.14
TAX_RATE: 0.30
```

## Funcionalidades Especiales

### Análisis de Sensibilidad
- Variación automática de ±10% en variables clave
- Gráfico tipo tornado mostrando impactos en NPV Equity
- Variables analizadas: CapEx, Precio energía, WACC, Factor de capacidad

### Dimensionamiento de Deuda
- Búsqueda binaria para encontrar máximo nivel de deuda
- Cumplimiento de DSCR objetivo (configurable)
- Optimización automática de estructura financiera

### Comparador Multi-Escenario
- Tabla consolidada con KPIs de todos los escenarios
- Ordenamiento por IRR Equity descendente
- Navegación rápida entre escenarios

## Notas Técnicas

### IRR con Fallback
La aplicación intenta usar `numpy_financial.irr`. Si no está disponible, implementa un fallback con búsqueda binaria en el rango [-0.99, 10.0].

### Validación de Parámetros
Los parámetros se validan contra rangos sugeridos. Los valores fuera de rango generan warnings pero son aceptados para permitir flexibilidad en el análisis.

### Caché de Resultados
Los resultados se almacenan en caché usando `st.cache_data` para acelerar la navegación entre escenarios y cambios menores en parámetros.

## Soporte y Contribuciones

Para reportar problemas o sugerir mejoras, por favor contactar al equipo de desarrollo.

## Licencia

Este proyecto es de uso interno y confidencial.

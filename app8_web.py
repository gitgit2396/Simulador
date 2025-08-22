"""
Aplicación web Streamlit para simulación económica de planta fotovoltaica
Incluye tres modos de ejecución: YAML, manual y multi-escenario
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_core import (
    DEFAULT_PARAMS, PARAM_RANGES, validate_parameters,
    run_financial_model, run_sensitivity_analysis, find_max_debt_by_dscr
)

# Configuración de página
st.set_page_config(
    page_title="Simulador Económico Planta Fotovoltaica",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache para resultados
@st.cache_data
def cached_run_model(params_tuple):
    """Cache de resultados del modelo"""
    # Convertir tuple de vuelta a dict para el modelo
    params_dict = dict(params_tuple)
    return run_financial_model(params_dict)

def format_number(value, decimals=2, is_percentage=False):
    """Formatea números con separador de miles"""
    if value is None:
        return "N/A"
    
    if is_percentage:
        return f"{value*100:,.{decimals}f}%"
    else:
        return f"{value:,.{decimals}f}"

def load_yaml_file(uploaded_file):
    """Carga parámetros desde archivo YAML"""
    try:
        content = yaml.safe_load(uploaded_file)
        return content, None
    except Exception as e:
        return None, f"Error al cargar YAML: {str(e)}"

def create_download_button(data, filename, file_format="csv"):
    """Crea botón de descarga en memoria"""
    if file_format == "csv":
        buffer = io.StringIO()
        data.to_csv(buffer, index=False)
        return st.download_button(
            label=f"📥 Descargar {filename}.csv",
            data=buffer.getvalue(),
            file_name=f"{filename}.csv",
            mime="text/csv"
        )
    elif file_format == "xlsx":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Datos')
        return st.download_button(
            label=f"📥 Descargar {filename}.xlsx",
            data=buffer.getvalue(),
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def render_parameter_form(initial_params=None):
    """Renderiza formulario de parámetros"""
    if initial_params is None:
        initial_params = DEFAULT_PARAMS
    
    params = {}
    
    st.subheader("📊 Parámetros del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Características Técnicas**")
        params['YEARS'] = st.number_input("Horizonte (años)", min_value=20, max_value=35, value=initial_params.get('YEARS', 30))
        params['PLANT_MW'] = st.number_input("Capacidad (MW)", min_value=1.0, max_value=300.0, value=initial_params.get('PLANT_MW', 50.0))
        params['CAP_FACTOR'] = st.number_input("Factor de Capacidad", min_value=0.18, max_value=0.33, value=initial_params.get('CAP_FACTOR', 0.25), format="%.3f")
        params['DEGRADATION'] = st.number_input("Degradación Anual", min_value=0.003, max_value=0.008, value=initial_params.get('DEGRADATION', 0.005), format="%.4f")
        
        st.markdown("**CapEx y Repotenciación**")
        params['CAPEX0_PER_KW'] = st.number_input("CapEx Inicial (USD/kW)", min_value=500.0, max_value=1200.0, value=initial_params.get('CAPEX0_PER_KW', 800.0))
        params['INVERTER_REPOWER_USD_PER_KW'] = st.number_input("Repotenciación (USD/kW)", min_value=20.0, max_value=80.0, value=initial_params.get('INVERTER_REPOWER_USD_PER_KW', 50.0))
        params['REP_YEAR'] = st.number_input("Año de Repotenciación", min_value=8, max_value=15, value=initial_params.get('REP_YEAR', 12))
        
        st.markdown("**Ingresos**")
        params['PRICE_USD_PER_MWH_Y1'] = st.number_input("Precio Año 1 (USD/MWh)", min_value=25.0, max_value=70.0, value=initial_params.get('PRICE_USD_PER_MWH_Y1', 45.0))
        params['PRICE_ESCALATION'] = st.number_input("Escalación Precio", min_value=0.00, max_value=0.03, value=initial_params.get('PRICE_ESCALATION', 0.015), format="%.3f")
    
    with col2:
        st.markdown("**Costos Operativos**")
        params['FOM_USD_PER_KW_YR'] = st.number_input("O&M Fijo (USD/kW/año)", min_value=8.0, max_value=25.0, value=initial_params.get('FOM_USD_PER_KW_YR', 15.0))
        params['VOM_USD_PER_MWH'] = st.number_input("O&M Variable (USD/MWh)", min_value=0.0, max_value=3.0, value=initial_params.get('VOM_USD_PER_MWH', 1.5))
        params['LAND_LEASE_USD_PER_MW_YR'] = st.number_input("Arriendo Terreno (USD/MW/año)", min_value=0.0, max_value=4000.0, value=initial_params.get('LAND_LEASE_USD_PER_MW_YR', 2000.0))
        params['OPEX_ESCALATION'] = st.number_input("Escalación OpEx", min_value=0.00, max_value=0.03, value=initial_params.get('OPEX_ESCALATION', 0.02), format="%.3f")
        
        st.markdown("**Financieros y Fiscales**")
        params['TAX_RATE'] = st.number_input("Tasa Impuesto", min_value=0.25, max_value=0.35, value=initial_params.get('TAX_RATE', 0.30), format="%.2f")
        params['WACC'] = st.number_input("WACC", min_value=0.07, max_value=0.13, value=initial_params.get('WACC', 0.10), format="%.3f")
        params['KE'] = st.number_input("Costo Equity (Ke)", min_value=0.12, max_value=0.20, value=initial_params.get('KE', 0.15), format="%.3f")
        params['NWC_PCT_OF_REVENUE'] = st.number_input("Capital Trabajo (% Ingresos)", min_value=0.00, max_value=0.15, value=initial_params.get('NWC_PCT_OF_REVENUE', 0.05), format="%.3f")
        
        st.markdown("**Estructura de Deuda**")
        params['DEBT_PCT_CAPEX0'] = st.number_input("Deuda (% CapEx)", min_value=0.0, max_value=0.60, value=initial_params.get('DEBT_PCT_CAPEX0', 0.30), format="%.2f")
        params['DEBT_RATE'] = st.number_input("Tasa Deuda", min_value=0.06, max_value=0.14, value=initial_params.get('DEBT_RATE', 0.08), format="%.3f")
        params['DEBT_TERM_YEARS'] = st.number_input("Plazo Deuda (años)", min_value=8, max_value=18, value=initial_params.get('DEBT_TERM_YEARS', 12))
        
        st.markdown("**Depreciación**")
        params['DEP_YEARS_CAPEX0'] = st.number_input("Años Depreciación CapEx", min_value=4, max_value=7, value=initial_params.get('DEP_YEARS_CAPEX0', 5))
        params['DEP_YEARS_REP'] = st.number_input("Años Depreciación Repotenciación", min_value=8, max_value=12, value=initial_params.get('DEP_YEARS_REP', 10))
    
    return params

def render_kpis(kpis, dscr_threshold=1.20):
    """Renderiza KPIs en formato de métricas"""
    st.subheader("📈 Indicadores Clave (KPIs)")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("NPV Proyecto (WACC)", format_number(kpis.get('NPV Proyecto (WACC)', 0), 0))
        irr_proy = kpis.get('IRR Proyecto')
        st.metric("IRR Proyecto", format_number(irr_proy, 2, True) if irr_proy is not None else "N/A")

    with col2:
        st.metric("NPV Equity (Ke)", format_number(kpis.get('NPV Equity (Ke)', 0), 0))
        irr_eq = kpis.get('IRR Equity')
        st.metric("IRR Equity", format_number(irr_eq, 2, True) if irr_eq is not None else "N/A")

    with col3:
        payback_proj = kpis.get('Payback Proyecto')
        st.metric("Payback Proyecto", f"{payback_proj:.1f} años" if payback_proj is not None else "N/A")
        payback_eq = kpis.get('Payback Equity')
        st.metric("Payback Equity", f"{payback_eq:.1f} años" if payback_eq is not None else "N/A")

    with col4:
        dscr_min = kpis.get('DSCR Mínimo')
        if dscr_min is not None:
            delta_color = "normal" if dscr_min >= dscr_threshold else "inverse"
            st.metric("DSCR Mínimo", f"{dscr_min:.2f}", delta=f"Umbral: {dscr_threshold:.2f}", delta_color=delta_color)
        else:
            st.metric("DSCR Mínimo", "N/A (Sin deuda)")

        dscr_avg = kpis.get('DSCR Promedio')
        st.metric("DSCR Promedio", f"{dscr_avg:.2f}" if dscr_avg is not None else "N/A")

def render_charts(df, scenario_name="Escenario"):
    """Renderiza gráficos del modelo"""
    st.subheader("📊 Gráficos")
    
    # Gráfico 1: Series financieras principales
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Energía Generada', 'Ingresos vs OpEx', 'EBITDA', 'Flujos de Caja'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    years = df['Año']
    
    # Energía
    fig1.add_trace(go.Scatter(x=years, y=df['Energía (MWh)'], name='Energía (MWh)', line=dict(color='orange')), row=1, col=1)
    
    # Ingresos vs OpEx
    fig1.add_trace(go.Scatter(x=years, y=df['Ingresos'], name='Ingresos', line=dict(color='green')), row=1, col=2)
    fig1.add_trace(go.Scatter(x=years, y=df['Opex Total'], name='OpEx Total', line=dict(color='red')), row=1, col=2)
    
    # EBITDA
    fig1.add_trace(go.Scatter(x=years, y=df['EBITDA'], name='EBITDA', line=dict(color='blue')), row=2, col=1)
    
    # Flujos de caja
    fig1.add_trace(go.Scatter(x=years, y=df['FCFF'], name='FCFF', line=dict(color='purple')), row=2, col=2)
    fig1.add_trace(go.Scatter(x=years, y=df['FCFE'], name='FCFE', line=dict(color='brown')), row=2, col=2)
    
    fig1.update_layout(height=600, title_text=f"Series Financieras - {scenario_name}")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: DSCR
    debt_mask = df['DSCR'] > 0
    if debt_mask.any():
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df.loc[debt_mask, 'Año'], 
            y=df.loc[debt_mask, 'DSCR'],
            mode='lines+markers',
            name='DSCR',
            line=dict(color='navy')
        ))
        fig2.add_hline(y=1.20, line_dash="dash", line_color="red", annotation_text="Umbral 1.20")
        fig2.update_layout(
            title=f"Cobertura de Servicio de Deuda (DSCR) - {scenario_name}",
            xaxis_title="Año",
            yaxis_title="DSCR",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

def main():
    """Función principal de la aplicación"""
    
    # Título principal
    st.title("☀️ Simulador Económico Planta Fotovoltaica")
    st.markdown("**Horizonte 30 años | Deuda | Depreciaciones | Repotenciación | Multi-escenario**")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    
    # Selector de modo
    mode = st.sidebar.selectbox(
        "Modo de Ejecución",
        [
            "Ejecutar con datos cargados (YAML)",
            "Ejecutar modificando parámetros",
            "Ejecutar todos los escenarios a la vez"
        ]
    )
    
    # Tag/Escenario
    scenario_tag = st.sidebar.text_input(
        "Tag/Escenario",
        placeholder="Dejar vacío para usar nombre del archivo"
    )
    
    # Toggle descargas
    enable_downloads = st.sidebar.toggle("Permitir descargas (opcional)", value=False)
    
    # Umbral DSCR
    dscr_threshold = st.sidebar.number_input(
        "Umbral DSCR de Referencia",
        min_value=1.0, max_value=2.0,
        value=1.20, step=0.05,
        format="%.2f"
    )
    
    # Variables para almacenar resultados
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    
    # Modo 1: YAML único
    if mode == "Ejecutar con datos cargados (YAML)":
        st.header("📁 Modo 1: Cargar Parámetros desde YAML")
        
        uploaded_file = st.file_uploader(
            "Subir archivo YAML",
            type=['yaml', 'yml'],
            help="Archivo con parámetros del modelo"
        )
        
        if uploaded_file:
            params, error = load_yaml_file(uploaded_file)
            
            if error:
                st.error(error)
            else:
                # Usar nombre del archivo si no hay tag
                if not scenario_tag:
                    scenario_tag = uploaded_file.name.split('.')[0]
                
                # Combinar con defaults
                final_params = DEFAULT_PARAMS.copy()
                final_params.update(params)
                
                # Validar parámetros
                warnings = validate_parameters(final_params)
                if warnings:
                    st.warning("⚠️ Algunos parámetros están fuera del rango sugerido:")
                    for param, warning in warnings.items():
                        st.write(f"- {param}: {warning}")
                
                # Ejecutar modelo
                with st.spinner("Ejecutando modelo..."):
                    result = cached_run_model(tuple(sorted(final_params.items())))
                    st.session_state.results[scenario_tag] = result
                    st.session_state.current_scenario = scenario_tag
                
                st.success(f"✅ Modelo ejecutado para escenario: {scenario_tag}")
    
    # Modo 2: Modificar parámetros
    elif mode == "Ejecutar modificando parámetros":
        st.header("📝 Modo 2: Modificar Parámetros")
        
        # Opción de cargar YAML base
        st.subheader("Parámetros Base (Opcional)")
        uploaded_base = st.file_uploader(
            "Cargar YAML base (opcional)",
            type=['yaml', 'yml'],
            help="Archivo base para precargar parámetros"
        )
        
        base_params = DEFAULT_PARAMS.copy()
        if uploaded_base:
            loaded_params, error = load_yaml_file(uploaded_base)
            if not error:
                base_params.update(loaded_params)
                st.info("✅ Parámetros base cargados desde YAML")
        
        # Formulario de parámetros
        params = render_parameter_form(base_params)
        
        # Botón ejecutar
        if st.button("🚀 Ejecutar Escenario", type="primary"):
            # Validar parámetros
            warnings = validate_parameters(params)
            if warnings:
                st.warning("⚠️ Algunos parámetros están fuera del rango sugerido:")
                for param, warning in warnings.items():
                    st.write(f"- {param}: {warning}")
            
            # Tag por defecto
            if not scenario_tag:
                scenario_tag = "manual"
            
            # Ejecutar modelo
            with st.spinner("Ejecutando modelo..."):
                result = cached_run_model(tuple(sorted(params.items())))
                st.session_state.results[scenario_tag] = result
                st.session_state.current_scenario = scenario_tag
            
            st.success(f"✅ Modelo ejecutado para escenario: {scenario_tag}")
        
        # Opción de guardar YAML
        if st.session_state.current_scenario and enable_downloads:
            st.subheader("💾 Guardar Configuración")
            yaml_content = yaml.dump(params, default_flow_style=False, allow_unicode=True)
            st.download_button(
                "📥 Descargar YAML",
                data=yaml_content,
                file_name=f"{scenario_tag}_params.yaml",
                mime="text/yaml"
            )
    
    # Modo 3: Multi-escenario
    elif mode == "Ejecutar todos los escenarios a la vez":
        st.header("📊 Modo 3: Multi-Escenario")
        
        uploaded_files = st.file_uploader(
            "Subir múltiples archivos YAML",
            type=['yaml', 'yml'],
            accept_multiple_files=True,
            help="Seleccionar varios archivos YAML para comparar escenarios"
        )
        
        if uploaded_files:
            if st.button("🚀 Ejecutar Todos los Escenarios", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Procesando {file.name}...")
                    
                    params, error = load_yaml_file(file)
                    if error:
                        st.error(f"Error en {file.name}: {error}")
                        continue
                    
                    # Combinar con defaults
                    final_params = DEFAULT_PARAMS.copy()
                    final_params.update(params)
                    
                    # Nombre del escenario
                    scenario_name = file.name.split('.')[0]
                    
                    # Ejecutar modelo
                    result = cached_run_model(tuple(sorted(final_params.items())))
                    st.session_state.results[scenario_name] = result
                    
                    # Actualizar progreso
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Todos los escenarios procesados")
                st.session_state.current_scenario = list(st.session_state.results.keys())[-1]
                st.success(f"✅ {len(uploaded_files)} escenarios ejecutados")
    
    # Mostrar resultados si existen
    if st.session_state.results:
        st.markdown("---")
        
        # Selector de escenario actual
        if len(st.session_state.results) > 1:
            st.subheader("🔄 Selector de Escenario")
            selected_scenario = st.selectbox(
                "Escenario a mostrar:",
                list(st.session_state.results.keys()),
                index=list(st.session_state.results.keys()).index(st.session_state.current_scenario) 
                if st.session_state.current_scenario in st.session_state.results else 0
            )
            st.session_state.current_scenario = selected_scenario
        else:
            selected_scenario = st.session_state.current_scenario
        
        # Obtener resultado actual
        current_result = st.session_state.results[selected_scenario]
        df = current_result['dataframe']
        kpis = current_result['kpis']
        debt_schedule = current_result['debt_schedule']
        parameters = current_result['parameters']
        
        # Tabs principales
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 KPIs", 
            "📋 Data", 
            "💰 Calendario Deuda", 
            "⚙️ Inputs", 
            "📊 Gráficos"
        ])
        
        with tab1:
            render_kpis(kpis, dscr_threshold)
            
            # Análisis adicionales
            with st.expander("🔍 Análisis de Sensibilidad"):
                if st.button("Ejecutar Sensibilidad ±10%"):
                    with st.spinner("Calculando sensibilidades..."):
                        sens_result = run_sensitivity_analysis(parameters)
                        sens_df = pd.DataFrame(sens_result['results'])
                        st.dataframe(sens_df, use_container_width=True)

            with st.expander("🎯 Dimensionamiento de Deuda por DSCR"):
                target_dscr = st.number_input(
                    "DSCR Objetivo",
                    min_value=1.10, max_value=2.00,
                    value=1.20, step=0.05
                )
                if st.button("Calcular Deuda Máxima"):
                    with st.spinner("Optimizando estructura de deuda..."):
                        debt_result = find_max_debt_by_dscr(parameters, target_dscr)
                        if debt_result['result']:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Deuda Óptima (% CapEx)", f"{debt_result['optimal_debt_pct']*100:.1f}%")
                            with col2:
                                st.metric("DSCR Logrado", f"{debt_result['achieved_dscr']:.2f}")
                            with col3:
                                opt_kpis = debt_result['result']['kpis']
                                st.metric("NPV Equity Optimizado", format_number(opt_kpis['NPV Equity (Ke)'], 0))
                        else:
                            st.warning("No se pudo encontrar una estructura de deuda viable")
        
        with tab2:
            st.subheader("📋 Datos Completos del Modelo")
            
            # Formatear números en el dataframe
            df_display = df.copy()
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Año':
                    df_display[col] = df_display[col].apply(lambda x: format_number(x, 2))
            
            st.dataframe(df_display, use_container_width=True)
            
            if enable_downloads:
                col1, col2 = st.columns(2)
                with col1:
                    create_download_button(df, f"{selected_scenario}_data", "csv")
                with col2:
                    create_download_button(df, f"{selected_scenario}_data", "xlsx")
        
        with tab3:
            st.subheader("💰 Calendario de Deuda")
            
            if not debt_schedule.empty:
                # Formatear números
                debt_display = debt_schedule.copy()
                numeric_cols = debt_display.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != 'Año':
                        debt_display[col] = debt_display[col].apply(lambda x: format_number(x, 2))
                
                st.dataframe(debt_display, use_container_width=True)
                
                if enable_downloads:
                    col1, col2 = st.columns(2)
                    with col1:
                        create_download_button(debt_schedule, f"{selected_scenario}_debt", "csv")
                    with col2:
                        create_download_button(debt_schedule, f"{selected_scenario}_debt", "xlsx")
            else:
                st.info("No hay estructura de deuda en este escenario")
        
        with tab4:
            st.subheader("⚙️ Parámetros Utilizados")
            
            # Mostrar parámetros en formato tabla
            params_df = pd.DataFrame([
                {"Parámetro": k, "Valor": v, "Unidad": ""}
                for k, v in parameters.items()
            ])
            
            st.dataframe(params_df, use_container_width=True)
            
            if enable_downloads:
                col1, col2 = st.columns(2)
                with col1:
                    create_download_button(params_df, f"{selected_scenario}_inputs", "csv")
                with col2:
                    # YAML download
                    yaml_content = yaml.dump(parameters, default_flow_style=False)
                    st.download_button(
                        "📥 Descargar YAML",
                        data=yaml_content,
                        file_name=f"{selected_scenario}_params.yaml",
                        mime="text/yaml"
                    )
        
        with tab5:
            render_charts(df, selected_scenario)
        
        # Tab adicional para comparador multi-escenario
        if len(st.session_state.results) > 1:
            st.markdown("---")
            st.subheader("📊 Comparador Multi-Escenario")
            
            # Crear tabla comparativa
            comparison_data = []
            for scenario_name, result in st.session_state.results.items():
                kpis = result['kpis']
                comparison_data.append({
                    'Escenario': scenario_name,
                    'IRR Equity': kpis.get('IRR Equity'),
                    'NPV Equity (Ke)': kpis.get('NPV Equity (Ke)'),
                    'NPV Proyecto (WACC)': kpis.get('NPV Proyecto (WACC)'),
                    'DSCR Mínimo': kpis.get('DSCR Mínimo'),
                    'DSCR Promedio': kpis.get('DSCR Promedio')
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Ordenar por IRR Equity descendente
            if 'IRR Equity' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('IRR Equity', ascending=False, na_position='last')
            
            # Formatear para display
            comparison_display = comparison_df.copy()
            for col in ['IRR Equity']:
                if col in comparison_display.columns:
                    comparison_display[col] = comparison_display[col].apply(lambda x: format_number(x, 2, True) if x is not None else "N/A")
            
            for col in ['NPV Equity (Ke)', 'NPV Proyecto (WACC)']:
                if col in comparison_display.columns:
                    comparison_display[col] = comparison_display[col].apply(lambda x: format_number(x, 0) if x is not None else "N/A")
            
            for col in ['DSCR Mínimo', 'DSCR Promedio']:
                if col in comparison_display.columns:
                    comparison_display[col] = comparison_display[col].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
            
            st.dataframe(comparison_display, use_container_width=True)
            
            # Selector para cambiar de escenario
            st.subheader("🔄 Navegar a Escenario")
            selected_for_nav = st.selectbox(
                "Seleccionar escenario para ver detalles:",
                list(st.session_state.results.keys()),
                key="nav_selector"
            )
            
            if st.button("Ver Detalles del Escenario"):
                st.session_state.current_scenario = selected_for_nav
                st.experimental_rerun()

if __name__ == "__main__":
    main()

"""
Módulo central para la simulación económica de planta fotovoltaica
Contiene toda la lógica de cálculo financiero de forma testeable
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Fallback para IRR si numpy_financial no está disponible
try:
    from numpy_financial import irr as np_irr
    HAS_NUMPY_FINANCIAL = True
except ImportError:
    HAS_NUMPY_FINANCIAL = False

# Parámetros por defecto con rangos de validación
DEFAULT_PARAMS = {
    'YEARS': 30,
    'PLANT_MW': 50.0,
    'CAPEX0_PER_KW': 800.0,
    'INVERTER_REPOWER_USD_PER_KW': 50.0,
    'CAP_FACTOR': 0.25,
    'DEGRADATION': 0.005,
    'PRICE_USD_PER_MWH_Y1': 45.0,
    'PRICE_ESCALATION': 0.015,
    'FOM_USD_PER_KW_YR': 15.0,
    'VOM_USD_PER_MWH': 1.5,
    'OPEX_ESCALATION': 0.02,
    'LAND_LEASE_USD_PER_MW_YR': 2000.0,
    'TAX_RATE': 0.30,
    'NWC_PCT_OF_REVENUE': 0.05,
    'WACC': 0.10,
    'KE': 0.15,
    'DEP_YEARS_CAPEX0': 5,
    'REP_YEAR': 12,
    'DEP_YEARS_REP': 10,
    'DEBT_PCT_CAPEX0': 0.30,
    'DEBT_RATE': 0.08,
    'DEBT_TERM_YEARS': 12
}

# Rangos de validación (min, max)
PARAM_RANGES = {
    'YEARS': (20, 35),
    'PLANT_MW': (1, 300),
    'CAPEX0_PER_KW': (500, 1200),
    'INVERTER_REPOWER_USD_PER_KW': (20, 80),
    'CAP_FACTOR': (0.18, 0.33),
    'DEGRADATION': (0.003, 0.008),
    'PRICE_USD_PER_MWH_Y1': (25, 70),
    'PRICE_ESCALATION': (0.00, 0.03),
    'FOM_USD_PER_KW_YR': (8, 25),
    'VOM_USD_PER_MWH': (0, 3),
    'OPEX_ESCALATION': (0.00, 0.03),
    'LAND_LEASE_USD_PER_MW_YR': (0, 4000),
    'TAX_RATE': (0.25, 0.35),
    'NWC_PCT_OF_REVENUE': (0.00, 0.15),
    'WACC': (0.07, 0.13),
    'KE': (0.12, 0.20),
    'DEP_YEARS_CAPEX0': (4, 7),
    'REP_YEAR': (8, 15),
    'DEP_YEARS_REP': (8, 12),
    'DEBT_PCT_CAPEX0': (0, 0.60),
    'DEBT_RATE': (0.06, 0.14),
    'DEBT_TERM_YEARS': (8, 18)
}

def validate_parameters(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Valida parámetros contra rangos esperados
    Retorna diccionario con warnings (no errores)
    """
    warnings = {}
    
    for param, value in params.items():
        if param in PARAM_RANGES:
            min_val, max_val = PARAM_RANGES[param]
            if value < min_val or value > max_val:
                warnings[param] = f"Valor {value} fuera del rango sugerido [{min_val}, {max_val}]"
    
    return warnings

def irr_fallback(cashflows: np.ndarray) -> float:
    """
    Cálculo de IRR con fallback usando búsqueda binaria
    """
    if HAS_NUMPY_FINANCIAL:
        try:
            result = np_irr(cashflows)
            return result if not np.isnan(result) else None
        except:
            pass
    
    # Fallback con búsqueda binaria
    def npv_at_rate(rate):
        return np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows)))
    
    # Búsqueda binaria en rango [-0.99, 10.0]
    low, high = -0.99, 10.0
    tolerance = 1e-6
    max_iterations = 100
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        npv_mid = npv_at_rate(mid)
        
        if abs(npv_mid) < tolerance:
            return mid
        elif npv_mid > 0:
            low = mid
        else:
            high = mid
    
    return None

def calculate_annuity_payment(principal: float, rate: float, periods: int) -> float:
    """
    Calcula cuota francesa (anualidad)
    """
    if rate == 0:
        return principal / periods
    return principal * (rate * (1 + rate)**periods) / ((1 + rate)**periods - 1)

def calculate_payback_period(cashflows: np.ndarray) -> Optional[float]:
    """
    Calcula período de payback con interpolación lineal
    """
    cumulative = np.cumsum(cashflows)
    
    # Buscar primer valor positivo
    positive_indices = np.where(cumulative > 0)[0]
    if len(positive_indices) == 0:
        return None
    
    first_positive = positive_indices[0]
    if first_positive == 0:
        return 0
    
    # Interpolación lineal
    cf_before = cumulative[first_positive - 1]
    cf_after = cumulative[first_positive]
    
    fraction = -cf_before / (cf_after - cf_before)
    return first_positive - 1 + fraction

def run_financial_model(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta el modelo financiero completo
    Retorna diccionario con resultados: dataframe, kpis, debt_schedule
    """
    
    # 1. Parámetros base
    years = params['YEARS']
    plant_mw = params['PLANT_MW']
    
    # 2. Cálculo de producción energética
    energy = np.zeros(years + 1)  # Año 0 = 0
    for t in range(1, years + 1):
        energy[t] = (plant_mw * 8760 * params['CAP_FACTOR'] * 
                    (1 - params['DEGRADATION'])**(t - 1))
    
    # 3. Precios e ingresos
    price = np.zeros(years + 1)
    revenue = np.zeros(years + 1)
    for t in range(1, years + 1):
        price[t] = (params['PRICE_USD_PER_MWH_Y1'] * 
                   (1 + params['PRICE_ESCALATION'])**(t - 1))
        revenue[t] = energy[t] * price[t]
    
    # 4. Costos O&M
    fom = np.zeros(years + 1)
    vom = np.zeros(years + 1)
    land_lease = np.zeros(years + 1)
    opex = np.zeros(years + 1)
    
    for t in range(1, years + 1):
        escalation_factor = (1 + params['OPEX_ESCALATION'])**(t - 1)
        fom[t] = (plant_mw * 1000) * params['FOM_USD_PER_KW_YR'] * escalation_factor
        vom[t] = energy[t] * params['VOM_USD_PER_MWH'] * escalation_factor
        land_lease[t] = plant_mw * params['LAND_LEASE_USD_PER_MW_YR'] * escalation_factor
        opex[t] = fom[t] + vom[t] + land_lease[t]
    
    # 5. CapEx y depreciación
    capex0 = (plant_mw * 1000) * params['CAPEX0_PER_KW']
    rep_cost = (plant_mw * 1000) * params['INVERTER_REPOWER_USD_PER_KW']
    
    capex = np.zeros(years + 1)
    capex[0] = capex0
    capex[params['REP_YEAR']] = rep_cost
    
    # Depreciación lineal
    depreciation = np.zeros(years + 1)
    
    # Depreciación CAPEX0
    annual_dep_capex0 = capex0 / params['DEP_YEARS_CAPEX0']
    for t in range(1, min(params['DEP_YEARS_CAPEX0'] + 1, years + 1)):
        depreciation[t] += annual_dep_capex0
    
    # Depreciación repotenciación
    annual_dep_rep = rep_cost / params['DEP_YEARS_REP']
    rep_start = params['REP_YEAR']
    rep_end = min(rep_start + params['DEP_YEARS_REP'], years + 1)
    for t in range(rep_start, rep_end):
        depreciation[t] += annual_dep_rep
    
    # 6. P&L
    ebitda = revenue - opex
    ebit = ebitda - depreciation
    tax = np.maximum(ebit, 0) * params['TAX_RATE']  # Solo sobre EBIT positivo
    nopat = ebit - tax
    
    # 7. Capital de trabajo
    nwc = params['NWC_PCT_OF_REVENUE'] * revenue
    delta_nwc = np.diff(nwc, prepend=0)
    
    # 8. FCFF (flujo libre sin apalancamiento)
    fcff = nopat + depreciation - capex - delta_nwc
    
    # 9. Estructura de deuda (cuota francesa)
    debt_balance = np.zeros(years + 1)
    interest = np.zeros(years + 1)
    principal = np.zeros(years + 1)
    debt_service = np.zeros(years + 1)
    new_debt = np.zeros(years + 1)
    
    debt0 = params['DEBT_PCT_CAPEX0'] * capex0
    debt_balance[0] = debt0
    new_debt[0] = debt0
    
    if debt0 > 0 and params['DEBT_TERM_YEARS'] > 0:
        annuity = calculate_annuity_payment(debt0, params['DEBT_RATE'], params['DEBT_TERM_YEARS'])
        
        for t in range(1, min(params['DEBT_TERM_YEARS'] + 1, years + 1)):
            interest[t] = debt_balance[t-1] * params['DEBT_RATE']
            principal[t] = annuity - interest[t]
            
            # Ajuste en último período si es necesario
            if t == params['DEBT_TERM_YEARS'] or t == years:
                principal[t] = debt_balance[t-1]
            
            debt_service[t] = interest[t] + principal[t]
            debt_balance[t] = debt_balance[t-1] - principal[t]
    
    # 10. DSCR
    dscr = np.divide(fcff, debt_service, out=np.zeros_like(fcff), where=debt_service>0)
    
    # 11. FCFE (flujo al equity)
    tax_shield = interest * params['TAX_RATE']
    fcfe = fcff - interest + tax_shield - principal + new_debt
    
    # 12. KPIs
    # Proyecto (unlevered)
    npv_project = np.sum(fcff / (1 + params['WACC'])**np.arange(years + 1))
    irr_project = irr_fallback(fcff)
    payback_project = calculate_payback_period(fcff)
    
    # Equity (levered)
    npv_equity = np.sum(fcfe / (1 + params['KE'])**np.arange(years + 1))
    irr_equity = irr_fallback(fcfe)
    payback_equity = calculate_payback_period(fcfe)
    
    # DSCR stats (solo durante años con deuda)
    debt_years_mask = debt_service > 0
    if np.any(debt_years_mask):
        dscr_min = np.min(dscr[debt_years_mask])
        dscr_avg = np.mean(dscr[debt_years_mask])
    else:
        dscr_min = dscr_avg = None
    
    # 13. DataFrame principal
    df = pd.DataFrame({
        'Año': np.arange(years + 1),
        'Energía (MWh)': energy,
        'Precio (USD/MWh)': price,
        'Ingresos': revenue,
        'O&M Fijo': fom,
        'O&M Variable': vom,
        'Arriendo': land_lease,
        'Opex Total': opex,
        'Depreciación': depreciation,
        'EBITDA': ebitda,
        'EBIT': ebit,
        'Impuesto': tax,
        'NOPAT': nopat,
        'CapEx': capex,
        'ΔNWC': delta_nwc,
        'FCFF': fcff,
        'Interés': interest,
        'Principal': principal,
        'Debt Service': debt_service,
        'DSCR': dscr,
        'Nueva Deuda': new_debt,
        'FCFE': fcfe,
        'Saldo Deuda': debt_balance
    })
    
    # 14. Calendario de deuda
    debt_schedule = df[['Año', 'Saldo Deuda', 'Interés', 'Principal', 'Debt Service']].copy()
    debt_schedule = debt_schedule[debt_schedule['Debt Service'] > 0]
    
    # 15. KPIs consolidados
    kpis = {
        'NPV Proyecto (WACC)': npv_project,
        'IRR Proyecto': irr_project,
        'Payback Proyecto': payback_project,
        'NPV Equity (Ke)': npv_equity,
        'IRR Equity': irr_equity,
        'Payback Equity': payback_equity,
        'DSCR Mínimo': dscr_min,
        'DSCR Promedio': dscr_avg
    }
    
    return {
        'dataframe': df,
        'kpis': kpis,
        'debt_schedule': debt_schedule,
        'parameters': params.copy()
    }

def run_sensitivity_analysis(base_params: Dict[str, Any], 
                           sensitivity_vars: list = None,
                           variation_pct: float = 0.10) -> Dict[str, Any]:
    """
    Ejecuta análisis de sensibilidad tipo tornado
    """
    if sensitivity_vars is None:
        sensitivity_vars = ['CAPEX0_PER_KW', 'PRICE_USD_PER_MWH_Y1', 'WACC', 'CAP_FACTOR']
    
    base_result = run_financial_model(base_params)
    base_npv_equity = base_result['kpis']['NPV Equity (Ke)']
    
    sensitivity_results = []
    
    for var in sensitivity_vars:
        if var not in base_params:
            continue
            
        base_value = base_params[var]
        
        # Caso optimista (+variation_pct)
        params_up = base_params.copy()
        params_up[var] = base_value * (1 + variation_pct)
        result_up = run_financial_model(params_up)
        npv_up = result_up['kpis']['NPV Equity (Ke)']
        
        # Caso pesimista (-variation_pct)
        params_down = base_params.copy()
        params_down[var] = base_value * (1 - variation_pct)
        result_down = run_financial_model(params_down)
        npv_down = result_down['kpis']['NPV Equity (Ke)']
        
        # Impacto
        impact_up = npv_up - base_npv_equity
        impact_down = npv_down - base_npv_equity
        total_range = abs(impact_up - impact_down)
        
        sensitivity_results.append({
            'Variable': var,
            'Valor Base': base_value,
            'NPV Base': base_npv_equity,
            'NPV +10%': npv_up,
            'NPV -10%': npv_down,
            'Impacto +10%': impact_up,
            'Impacto -10%': impact_down,
            'Rango Total': total_range
        })
    
    # Ordenar por rango total (mayor impacto primero)
    sensitivity_results.sort(key=lambda x: x['Rango Total'], reverse=True)
    
    return {
        'base_npv': base_npv_equity,
        'results': sensitivity_results
    }

def find_max_debt_by_dscr(base_params: Dict[str, Any], 
                         target_dscr: float = 1.20,
                         tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Encuentra el máximo nivel de deuda que cumple un DSCR objetivo
    usando búsqueda binaria
    """
    low_debt = 0.0
    high_debt = 0.80  # Máximo 80%
    max_iterations = 50
    
    best_result = None
    
    for _ in range(max_iterations):
        mid_debt = (low_debt + high_debt) / 2
        
        test_params = base_params.copy()
        test_params['DEBT_PCT_CAPEX0'] = mid_debt
        
        result = run_financial_model(test_params)
        min_dscr = result['kpis']['DSCR Mínimo']
        
        if min_dscr is None:  # Sin deuda
            low_debt = mid_debt
            continue
            
        if abs(min_dscr - target_dscr) < tolerance:
            best_result = result
            break
        elif min_dscr > target_dscr:
            low_debt = mid_debt
            best_result = result
        else:
            high_debt = mid_debt
    
    return {
        'optimal_debt_pct': mid_debt if best_result else 0,
        'achieved_dscr': min_dscr if best_result else None,
        'result': best_result
    }

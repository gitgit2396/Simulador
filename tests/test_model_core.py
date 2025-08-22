import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from model_core import (
    run_financial_model,
    run_sensitivity_analysis,
    find_max_debt_by_dscr,
    DEFAULT_PARAMS,
)


def test_run_financial_model_basic_structure():
    params = DEFAULT_PARAMS.copy()
    result = run_financial_model(params)
    assert 'dataframe' in result and 'kpis' in result
    assert result['kpis']['DSCR Mínimo'] is not None


def test_run_financial_model_no_debt():
    params = DEFAULT_PARAMS.copy()
    params['DEBT_PCT_CAPEX0'] = 0.0
    result = run_financial_model(params)
    assert result['kpis']['DSCR Mínimo'] is None
    assert result['debt_schedule'].empty


def test_run_financial_model_low_dscr():
    params = DEFAULT_PARAMS.copy()
    params['DEBT_PCT_CAPEX0'] = 0.6
    result = run_financial_model(params)
    assert result['kpis']['DSCR Mínimo'] < 1


def test_run_financial_model_extreme_degradation():
    params = DEFAULT_PARAMS.copy()
    params['DEGRADATION'] = 0.5
    result = run_financial_model(params)
    df = result['dataframe']
    assert df.loc[1, 'Energía (MWh)'] > df.loc[2, 'Energía (MWh)']


def test_run_sensitivity_analysis_structure():
    params = DEFAULT_PARAMS.copy()
    res = run_sensitivity_analysis(params, sensitivity_vars=['CAPEX0_PER_KW', 'WACC'])
    base_npv = run_financial_model(params)['kpis']['NPV Equity (Ke)']
    assert res['base_npv'] == base_npv
    assert len(res['results']) == 2


def test_find_max_debt_by_dscr_target():
    params = DEFAULT_PARAMS.copy()
    res = find_max_debt_by_dscr(params, target_dscr=1.2)
    assert res['achieved_dscr'] is not None
    assert abs(res['achieved_dscr'] - 1.2) < 0.05
    assert res['optimal_debt_pct'] > 0


def test_find_max_debt_by_dscr_low_target():
    params = DEFAULT_PARAMS.copy()
    res = find_max_debt_by_dscr(params, target_dscr=0.1)
    assert res['optimal_debt_pct'] > 0.79
    assert res['achieved_dscr'] < 0.2

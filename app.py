import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model


st.set_page_config(
    page_title="MILP-AHP | Modelo Word",
    page_icon="🧪",
    layout="wide",
)

# Mantiene la última solución aunque Streamlit haga rerun al descargar archivos.
# Esto evita que el resultado desaparezca al presionar un botón de descarga.
if "last_solution_bundle" not in st.session_state:
    st.session_state.last_solution_bundle = None

# =========================================================
# CONFIGURACIÓN ESTRICTA SEGÚN FORMULACIÓN WORD
# =========================================================
# Conjuntos:
# I = círculos; S = separaciones; F = actividades de campo;
# K = análisis de laboratorio; W = {Dia, Noche}
#
# Esta versión elimina elementos no formulados en el Word:
# - no usa mínimos por actividad/análisis
# - no pregunta qué actividad produce muestra de laboratorio
# - no usa reserva/imprevistos
# - no usa selector floor/ceil/round: aplica n_i,s = max(1, floor(P_i/s))
# - no permite cambiar la lógica de activación del círculo
# - conservación de muestras: sum_k L_k <= sum_f Q_f
# =========================================================

SHIFTS = ["Dia", "Noche"]
W_SCALE = 10_000


@dataclass
class ParsedData:
    params: Dict[str, float]
    circles: pd.DataFrame
    separations: pd.DataFrame
    field_df: pd.DataFrame
    lab_df: pd.DataFrame


PARAM_DEFAULTS = [
    {"Parametro": "T_max", "Descripcion": "Días totales disponibles para el muestreo", "Unidad": "dias", "Valor": 60},
    {"Parametro": "H_turno", "Descripcion": "Horas efectivas de trabajo por turno", "Unidad": "h/turno", "Valor": 8},
    {"Parametro": "C_total", "Descripcion": "Presupuesto total disponible", "Unidad": "COP", "Valor": 446852500},
    {"Parametro": "E_max", "Descripcion": "Número máximo de equipos de trabajo disponibles por turno", "Unidad": "equipos", "Valor": 2},
    {"Parametro": "beta_F", "Descripcion": "Peso AHP de primer nivel del bloque campo", "Unidad": "proporcion", "Valor": 0.6420},
    {"Parametro": "beta_L", "Descripcion": "Peso AHP de primer nivel del bloque laboratorio", "Unidad": "proporcion", "Valor": 0.3580},
]

DEFAULT_CIRCLES = [
    {"ID": "C1", "Perimetro_m": 113.0},
    {"ID": "C2", "Perimetro_m": 164.0},
    {"ID": "C3", "Perimetro_m": 702.0},
]

DEFAULT_SEPARATIONS = [
    {"separacion_m": 10},
    {"separacion_m": 20},
    {"separacion_m": 50},
]

DEFAULT_FIELD_ROWS = [
    {"actividad": "Muestreo puntual estándar", "alpha_f": 0.3758, "t_Dia_h": 1, "t_Noche_h": 0, "c_h_Dia_COP_h": 189125, "c_h_Noche_COP_h": 250000},
    {"actividad": "Muestreo multinivel", "alpha_f": 0.3154, "t_Dia_h": 3, "t_Noche_h": 0, "c_h_Dia_COP_h": 189125, "c_h_Noche_COP_h": 250000},
    {"actividad": "Medición 24 horas", "alpha_f": 0.2268, "t_Dia_h": 12, "t_Noche_h": 13, "c_h_Dia_COP_h": 189125, "c_h_Noche_COP_h": 250000},
    {"actividad": "Medición extendida", "alpha_f": 0.0820, "t_Dia_h": 60, "t_Noche_h": 61, "c_h_Dia_COP_h": 189125, "c_h_Noche_COP_h": 250000},
]

DEFAULT_LAB_ROWS = [
    {"analisis": "Cromatografía de Gas", "gamma_k": 0.4221, "c_lab_COP_muestra": 1190000},
    {"analisis": "Isotopía de Deuterio", "gamma_k": 0.2506, "c_lab_COP_muestra": 6538812},
    {"analisis": "Isotopía de Helio", "gamma_k": 0.1223, "c_lab_COP_muestra": 6538812},
    {"analisis": "Caracterización mineralógica", "gamma_k": 0.1115, "c_lab_COP_muestra": 1953385},
    {"analisis": "Biogeoquímica", "gamma_k": 0.0935, "c_lab_COP_muestra": 3015249},
]


# -----------------------------
# Utilidades de datos
# -----------------------------

def default_manual_data() -> ParsedData:
    params = {row["Parametro"]: row["Valor"] for row in PARAM_DEFAULTS}
    return ParsedData(
        params=params,
        circles=pd.DataFrame(DEFAULT_CIRCLES),
        separations=pd.DataFrame(DEFAULT_SEPARATIONS),
        field_df=pd.DataFrame(DEFAULT_FIELD_ROWS),
        lab_df=pd.DataFrame(DEFAULT_LAB_ROWS),
    )


def build_template_excel_bytes() -> bytes:
    """Plantilla descargable estricta según parámetros y conjuntos del Word."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(PARAM_DEFAULTS).to_excel(writer, sheet_name="Parametros", index=False)
        pd.DataFrame(DEFAULT_CIRCLES).to_excel(writer, sheet_name="Circulos", index=False)
        pd.DataFrame(DEFAULT_SEPARATIONS).to_excel(writer, sheet_name="Separaciones", index=False)
        pd.DataFrame(DEFAULT_FIELD_ROWS).to_excel(writer, sheet_name="Campo", index=False)
        pd.DataFrame(DEFAULT_LAB_ROWS).to_excel(writer, sheet_name="Laboratorio", index=False)
        notes = pd.DataFrame([
            {"Hoja": "Parametros", "Descripcion": "Debe conservar: Parametro, Descripcion, Unidad, Valor. Incluye T_max, H_turno, C_total, E_max, beta_F, beta_L."},
            {"Hoja": "Circulos", "Descripcion": "Conjunto I. Columnas: ID, Perimetro_m."},
            {"Hoja": "Separaciones", "Descripcion": "Conjunto S. Columna: separacion_m."},
            {"Hoja": "Campo", "Descripcion": "Conjunto F. Columnas: actividad, alpha_f, t_Dia_h, t_Noche_h, c_h_Dia_COP_h, c_h_Noche_COP_h. No hay mínimos ni produce_muestra_lab."},
            {"Hoja": "Laboratorio", "Descripcion": "Conjunto K. Columnas: analisis, gamma_k, c_lab_COP_muestra."},
            {"Hoja": "Modelo", "Descripcion": "Conservación: sum_k L_k <= sum_f Q_f. Todas las actividades de campo aportan a sum_f Q_f."},
        ])
        notes.to_excel(writer, sheet_name="Notas_formato", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                max_len = 10
                for cell in column_cells[:150]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[column_cells[0].column_letter].width = min(max_len + 2, 65)
    return output.getvalue()


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clean_name_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
    )


def _read_params(df: pd.DataFrame) -> Dict[str, float]:
    if "Parametro" not in df.columns or "Valor" not in df.columns:
        # Acepta también las primeras 4 columnas de la plantilla.
        if len(df.columns) >= 4:
            tmp = df.iloc[:, :4].copy()
            tmp.columns = ["Parametro", "Descripcion", "Unidad", "Valor"]
            df = tmp
        else:
            raise ValueError("La hoja Parametros debe contener las columnas Parametro y Valor.")
    tmp = df[["Parametro", "Valor"]].dropna(subset=["Parametro"]).copy()
    tmp["Parametro"] = _clean_name_series(tmp["Parametro"])
    return dict(zip(tmp["Parametro"], tmp["Valor"]))


def parse_excel(file_obj) -> ParsedData:
    xls = pd.ExcelFile(file_obj)
    sheet_map = {s.lower().strip(): s for s in xls.sheet_names}
    required = {"parametros", "circulos", "separaciones", "campo", "laboratorio"}
    if not required.issubset(set(sheet_map)):
        raise ValueError(
            "Formato no reconocido. Usa la plantilla estricta con hojas: "
            "Parametros, Circulos, Separaciones, Campo y Laboratorio."
        )

    params = _read_params(xls.parse(sheet_map["parametros"]))

    circles = xls.parse(sheet_map["circulos"])
    circles = circles.iloc[:, :2].copy()
    circles.columns = ["ID", "Perimetro_m"]

    separations = xls.parse(sheet_map["separaciones"])
    separations = separations.iloc[:, :1].copy()
    separations.columns = ["separacion_m"]

    field_df = xls.parse(sheet_map["campo"])
    if len(field_df.columns) < 6:
        raise ValueError("La hoja Campo debe tener columnas: actividad, alpha_f, t_Dia_h, t_Noche_h, c_h_Dia_COP_h, c_h_Noche_COP_h.")
    field_df = field_df.iloc[:, :6].copy()
    field_df.columns = ["actividad", "alpha_f", "t_Dia_h", "t_Noche_h", "c_h_Dia_COP_h", "c_h_Noche_COP_h"]

    lab_df = xls.parse(sheet_map["laboratorio"])
    if len(lab_df.columns) < 3:
        raise ValueError("La hoja Laboratorio debe tener columnas: analisis, gamma_k, c_lab_COP_muestra.")
    lab_df = lab_df.iloc[:, :3].copy()
    lab_df.columns = ["analisis", "gamma_k", "c_lab_COP_muestra"]

    return ParsedData(params=params, circles=circles, separations=separations, field_df=field_df, lab_df=lab_df)


def clean_model_inputs(
    circles: pd.DataFrame,
    separations: pd.DataFrame,
    field_df: pd.DataFrame,
    lab_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int], pd.DataFrame, pd.DataFrame]:
    circles = circles.copy()
    separations = separations.copy()
    field_df = field_df.copy()
    lab_df = lab_df.copy()

    if "ID" not in circles.columns or "Perimetro_m" not in circles.columns:
        raise ValueError("La tabla de círculos debe contener ID y Perimetro_m.")
    circles["ID"] = _clean_name_series(circles["ID"])
    circles["Perimetro_m"] = pd.to_numeric(circles["Perimetro_m"], errors="coerce")
    circles = circles[(circles["ID"] != "") & circles["Perimetro_m"].notna() & (circles["Perimetro_m"] > 0)]
    circles = circles.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)

    if "separacion_m" not in separations.columns:
        raise ValueError("La tabla de separaciones debe contener separacion_m.")
    sep = pd.to_numeric(separations["separacion_m"], errors="coerce").dropna()
    sep = sorted(set(int(round(x)) for x in sep if x > 0))

    required_field = ["actividad", "alpha_f", "t_Dia_h", "t_Noche_h", "c_h_Dia_COP_h", "c_h_Noche_COP_h"]
    for col in required_field:
        if col not in field_df.columns:
            raise ValueError(f"La tabla Campo debe contener la columna {col}.")
    field_df["actividad"] = _clean_name_series(field_df["actividad"])
    field_df = field_df[field_df["actividad"] != ""].copy()
    for col in ["alpha_f", "t_Dia_h", "t_Noche_h", "c_h_Dia_COP_h", "c_h_Noche_COP_h"]:
        field_df[col] = pd.to_numeric(field_df[col], errors="coerce").fillna(0)
    field_df = field_df[(field_df["t_Dia_h"] + field_df["t_Noche_h"]) > 0].reset_index(drop=True)

    required_lab = ["analisis", "gamma_k", "c_lab_COP_muestra"]
    for col in required_lab:
        if col not in lab_df.columns:
            raise ValueError(f"La tabla Laboratorio debe contener la columna {col}.")
    lab_df["analisis"] = _clean_name_series(lab_df["analisis"])
    lab_df = lab_df[lab_df["analisis"] != ""].copy()
    for col in ["gamma_k", "c_lab_COP_muestra"]:
        lab_df[col] = pd.to_numeric(lab_df[col], errors="coerce").fillna(0)
    lab_df = lab_df[lab_df["c_lab_COP_muestra"] >= 0].reset_index(drop=True)

    if circles.empty:
        raise ValueError("No hay círculos válidos.")
    if not sep:
        raise ValueError("Debes introducir al menos una separación positiva.")
    if field_df.empty:
        raise ValueError("No hay actividades de campo válidas.")
    if lab_df.empty:
        raise ValueError("No hay análisis de laboratorio válidos.")
    if field_df["alpha_f"].sum() <= 0:
        raise ValueError("La suma de alpha_f debe ser mayor que cero.")
    if lab_df["gamma_k"].sum() <= 0:
        raise ValueError("La suma de gamma_k debe ser mayor que cero.")
    return circles, sep, field_df, lab_df


def validate_weight_sum(name: str, value: float, tolerance: float = 0.02):
    if abs(float(value) - 1.0) > tolerance:
        raise ValueError(f"Según la formulación, {name} debe sumar 1. Valor actual: {value:.4f}.")


def compute_n_points(perimeter: float, separation: int) -> int:
    """n_i,s = max(1, floor(P_i/s)), tal como aparece en la formulación Word."""
    if separation <= 0:
        raise ValueError("Todas las separaciones deben ser mayores que cero.")
    if not math.isfinite(float(perimeter)) or perimeter <= 0:
        raise ValueError(f"Perímetro inválido: {perimeter}.")
    return max(1, int(math.floor(float(perimeter) / int(separation))))


# -----------------------------
# Reporte Excel
# -----------------------------

INVALID_SHEET_CHARS = str.maketrans({c: "_" for c in '[]:*?/\\'})


def safe_sheet_name(name: str, prefix: str = "", used=None) -> str:
    used = used if used is not None else set()
    base = f"{prefix}{name}".translate(INVALID_SHEET_CHARS).strip()
    base = "_".join(base.split()) or "Hoja"
    base = base[:31]
    candidate = base
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{base[:31-len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _integer_proportional_allocation(total: int, capacities: pd.Series) -> pd.Series:
    total = int(total)
    capacities = pd.to_numeric(capacities, errors="coerce").fillna(0).astype(float)
    result = pd.Series([0] * len(capacities), index=capacities.index, dtype=int)
    if total <= 0 or capacities.sum() <= 0:
        return result
    # Como la restricción del Word es L_k <= sum_f Q_f, cada análisis k se reparte
    # sobre la disponibilidad total de campo sin consumirla globalmente frente a otros k.
    raw = capacities / capacities.sum() * total
    base = raw.apply(math.floor).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = (raw - base).sort_values(ascending=False).index[:remainder]
        base.loc[order] += 1
    return base.astype(int)


def build_lab_allocation_by_circle(plan_df: pd.DataFrame, lab_result_df: pd.DataFrame, lab_input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if plan_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    all_circles = plan_df[["circulo", "perimetro_m"]].drop_duplicates().sort_values("circulo").reset_index(drop=True)
    capacity = plan_df.groupby("circulo", as_index=True)["puntos"].sum().rename("muestras_campo_total")
    wide = all_circles.merge(capacity, left_on="circulo", right_index=True, how="left")
    wide["muestras_campo_total"] = pd.to_numeric(wide["muestras_campo_total"], errors="coerce").fillna(0).astype(int)

    lab_costs = lab_input_df[["analisis", "c_lab_COP_muestra"]].copy()
    lab_counts = lab_result_df[["analisis", "muestras_L"]].merge(lab_costs, on="analisis", how="left")
    long_rows = []
    for _, lab_row in lab_counts.iterrows():
        analysis = lab_row["analisis"]
        total_samples = int(lab_row["muestras_L"])
        cost_per_sample = _safe_float(lab_row.get("c_lab_COP_muestra"), 0)
        allocation = _integer_proportional_allocation(total_samples, wide["muestras_campo_total"])
        wide[analysis] = allocation.values
        for idx, alloc in allocation.items():
            long_rows.append({
                "circulo": wide.loc[idx, "circulo"],
                "perimetro_m": wide.loc[idx, "perimetro_m"],
                "muestras_campo_total": int(wide.loc[idx, "muestras_campo_total"]),
                "analisis": analysis,
                "muestras_asignadas": int(alloc),
                "costo_estimado_COP": int(round(int(alloc) * cost_per_sample)),
            })
    wide["analisis_totales_asignados"] = wide[[c for c in wide.columns if c not in ["circulo", "perimetro_m", "muestras_campo_total"]]].sum(axis=1)
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = long_df.sort_values(["circulo", "analisis"]).reset_index(drop=True)
    return wide, long_df


def model_equations_table() -> pd.DataFrame:
    rows = [
        ("Conjuntos", "I, S, F, K, W={Dia,Noche}"),
        ("Puntos", "n_i,s = max(1, floor(P_i / s))"),
        ("Activación", "sum_s y_i,f,s = z_i,f  para todo i,f"),
        ("Integralidad", "sum_f z_i,f >= 1  para todo i"),
        ("Cantidad campo", "Q_f = sum_i sum_s n_i,s * y_i,f,s"),
        ("Costo campo", "C_f = sum_w c_h,f,w * t_f,w * Q_f"),
        ("Costo lab", "C_k = c_lab,k * L_k"),
        ("Agregados", "C_field=sum_f C_f; C_lab=sum_k C_k; C_used=C_field+C_lab"),
        ("Presupuesto", "C_used <= C_total"),
        ("Capacidad", "sum_f t_f,w * Q_f <= E_w * T_max * H_turno  para todo w"),
        ("Equipos", "E_w <= E_max  para todo w"),
        ("Conservación", "sum_k L_k <= sum_f Q_f"),
        ("AHP global", "C_field+eF- - eF+ = beta_F*C_used; C_lab+eL- - eL+ = beta_L*C_used"),
        ("AHP campo", "C_f+d_f- - d_f+ = alpha_f*C_field"),
        ("AHP lab", "C_k+u_k- - u_k+ = gamma_k*C_lab  para todo k"),
        ("Etapa 1", "min Z1 = eF-+eF+ + eL-+eL+"),
        ("Etapa 2", "min Z2 = sum_f(d_f-+d_f+) + sum_k(u_k-+u_k+) sujeto a Z1=Z1*"),
        ("Etapa 3", "max C_used sujeto a Z1=Z1* y Z2=Z2*"),
    ]
    return pd.DataFrame(rows, columns=["Bloque", "Ecuación implementada"])


def build_excel_report(result: Dict, field_input_df: pd.DataFrame, lab_input_df: pd.DataFrame, params_used: Dict) -> bytes:
    output = io.BytesIO()
    used_sheets = set()
    kpis = result.get("kpis", {})
    summary = pd.DataFrame([{"Indicador": k, "Valor": v} for k, v in kpis.items()])
    plan = result.get("plan", pd.DataFrame()).copy()
    field = result.get("field", pd.DataFrame()).copy()
    lab = result.get("lab", pd.DataFrame()).copy()
    logs = pd.DataFrame(result.get("logs", []))
    lab_by_circle, lab_long = build_lab_allocation_by_circle(plan, lab, lab_input_df)
    params_df = pd.DataFrame([{"Parametro": k, "Valor": v} for k, v in params_used.items()])
    equations = model_equations_table()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sheet = safe_sheet_name("Resumen", used=used_sheets)
        summary.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
        field.to_excel(writer, sheet_name=sheet, index=False, startrow=len(summary) + 3)
        lab.to_excel(writer, sheet_name=sheet, index=False, startrow=len(summary) + len(field) + 6)

        for name, df in [
            ("Parametros_usados", params_df),
            ("Modelo_usado", equations),
            ("Plan_completo", plan),
            ("Campo_resumen", field),
            ("Laboratorio_resumen", lab),
            ("Lab_por_circulo", lab_by_circle),
            ("Lab_detalle_circulo", lab_long),
        ]:
            sheet = safe_sheet_name(name, used=used_sheets)
            df.to_excel(writer, sheet_name=sheet, index=False)

        if not plan.empty and "actividad" in plan.columns:
            for activity in sorted(plan["actividad"].dropna().unique()):
                filtered = plan[plan["actividad"] == activity].copy()
                sheet = safe_sheet_name(str(activity), prefix="Campo_", used=used_sheets)
                filtered.to_excel(writer, sheet_name=sheet, index=False)

        if not lab_long.empty and "analisis" in lab_long.columns:
            for analysis in sorted(lab_long["analisis"].dropna().unique()):
                filtered = lab_long[lab_long["analisis"] == analysis].copy()
                sheet = safe_sheet_name(str(analysis), prefix="Lab_", used=used_sheets)
                filtered.to_excel(writer, sheet_name=sheet, index=False)

        sheet = safe_sheet_name("Log_solver", used=used_sheets)
        logs.to_excel(writer, sheet_name=sheet, index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                header = col_cells[0].value if col_cells else ""
                max_len = len(str(header)) if header is not None else 0
                for cell in col_cells[:250]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 55)
    return output.getvalue()


# -----------------------------
# Solver CP-SAT
# -----------------------------

def solve_milp_ahp_word(
    circles: pd.DataFrame,
    separations: List[int],
    field_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    budget_cop: int,
    days_available: int,
    hours_per_shift: int,
    max_teams: int,
    beta_field: float,
    beta_lab: float,
    time_limit_per_stage: int,
) -> Dict:
    circles, separations, field_df, lab_df = clean_model_inputs(circles, pd.DataFrame({"separacion_m": separations}), field_df, lab_df)

    validate_weight_sum("beta_F + beta_L", beta_field + beta_lab)
    validate_weight_sum("sum_f alpha_f", float(field_df["alpha_f"].sum()))
    validate_weight_sum("sum_k gamma_k", float(lab_df["gamma_k"].sum()))

    I = circles["ID"].tolist()
    F = field_df["actividad"].tolist()
    K = lab_df["analisis"].tolist()
    W = SHIFTS
    perimeter = dict(zip(I, circles["Perimetro_m"].astype(float)))

    budget = int(round(budget_cop))
    beta = {"Campo": int(round(beta_field * W_SCALE)), "Laboratorio": int(round(beta_lab * W_SCALE))}
    alpha = {row["actividad"]: int(round(row["alpha_f"] * W_SCALE)) for _, row in field_df.iterrows()}
    gamma = {row["analisis"]: int(round(row["gamma_k"] * W_SCALE)) for _, row in lab_df.iterrows()}

    t = {}
    c_h = {}
    for _, row in field_df.iterrows():
        f = row["actividad"]
        t[(f, "Dia")] = int(round(float(row["t_Dia_h"])))
        t[(f, "Noche")] = int(round(float(row["t_Noche_h"])))
        c_h[(f, "Dia")] = int(round(float(row["c_h_Dia_COP_h"])))
        c_h[(f, "Noche")] = int(round(float(row["c_h_Noche_COP_h"])))
    lab_cost = dict(zip(K, lab_df["c_lab_COP_muestra"].round().astype(int)))
    n = {(i, s): compute_n_points(perimeter[i], s) for i in I for s in separations}

    model = cp_model.CpModel()

    # Variables
    z = {(i, f): model.NewBoolVar(f"z__{i}__{f}") for i in I for f in F}
    y = {(i, f, s): model.NewBoolVar(f"y__{i}__{f}__{s}") for i in I for f in F for s in separations}
    max_q = sum(max(n[(i, s)] for s in separations) for i in I)
    Q = {f: model.NewIntVar(0, max_q, f"Q__{f}") for f in F}
    Cf = {f: model.NewIntVar(0, budget, f"Cf__{f}") for f in F}

    min_lab_cost = max(1, min([v for v in lab_cost.values() if v > 0] or [1]))
    lab_upper_bound = int(budget / min_lab_cost) + max_q + 100
    L = {k: model.NewIntVar(0, lab_upper_bound, f"L__{k}") for k in K}
    Ck = {k: model.NewIntVar(0, budget, f"Ck__{k}") for k in K}

    E = {w: model.NewIntVar(0, max_teams, f"E__{w}") for w in W}
    Cfield = model.NewIntVar(0, budget, "Cfield")
    Clab = model.NewIntVar(0, budget, "Clab")
    Cused = model.NewIntVar(0, budget, "Cused")

    # 4.1 Definición geométrica y activación: sum_s y_i,f,s = z_i,f
    for i in I:
        for f in F:
            model.Add(sum(y[(i, f, s)] for s in separations) == z[(i, f)])
        # 4.2 Integralidad: cada círculo tiene al menos una actividad
        model.Add(sum(z[(i, f)] for f in F) >= 1)

    # 4.3 Cantidades de campo
    for f in F:
        model.Add(Q[f] == sum(n[(i, s)] * y[(i, f, s)] for i in I for s in separations))
        # 4.4 Costos de campo: C_f = sum_w c_h,w * t_f,w * Q_f
        cost_per_unit = sum(c_h[(f, w)] * t[(f, w)] for w in W)
        model.Add(Cf[f] == int(cost_per_unit) * Q[f])

    # 4.5 Costos de laboratorio
    for k in K:
        model.Add(Ck[k] == lab_cost[k] * L[k])

    # 4.6 Agregados y 4.7 presupuesto
    model.Add(Cfield == sum(Cf[f] for f in F))
    model.Add(Clab == sum(Ck[k] for k in K))
    model.Add(Cused == Cfield + Clab)
    model.Add(Cused <= budget)

    # 4.8 Capacidad operativa por turno: sum_f t_f,w Q_f <= E_w*Tmax*H_turno
    for w in W:
        model.Add(sum(t[(f, w)] * Q[f] for f in F) <= E[w] * int(days_available) * int(hours_per_shift))
        # 4.9 límite de equipos
        model.Add(E[w] <= int(max_teams))

    # 4.13 Conservación de muestras: el TOTAL de análisis de laboratorio
    # no puede exceder el TOTAL de muestras de campo recolectadas.
    # Todas las actividades de campo aportan a sum_f Q_f.
    total_field_samples = sum(Q[f] for f in F)
    model.Add(sum(L[k] for k in K) <= total_field_samples)

    # Desviaciones AHP con variables explícitas, como en el Word.
    max_dev = max(1, budget * W_SCALE)
    eF_minus = model.NewIntVar(0, max_dev, "eF_minus")
    eF_plus = model.NewIntVar(0, max_dev, "eF_plus")
    eL_minus = model.NewIntVar(0, max_dev, "eL_minus")
    eL_plus = model.NewIntVar(0, max_dev, "eL_plus")
    model.Add(W_SCALE * Cfield + eF_minus - eF_plus == beta["Campo"] * Cused)
    model.Add(W_SCALE * Clab + eL_minus - eL_plus == beta["Laboratorio"] * Cused)
    Z1 = model.NewIntVar(0, 4 * max_dev, "Z1")
    model.Add(Z1 == eF_minus + eF_plus + eL_minus + eL_plus)

    d_minus, d_plus = {}, {}
    u_minus, u_plus = {}, {}
    field_dev_terms = []
    lab_dev_terms = []
    for f in F:
        d_minus[f] = model.NewIntVar(0, max_dev, f"d_minus__{f}")
        d_plus[f] = model.NewIntVar(0, max_dev, f"d_plus__{f}")
        model.Add(W_SCALE * Cf[f] + d_minus[f] - d_plus[f] == alpha[f] * Cfield)
        field_dev_terms.extend([d_minus[f], d_plus[f]])
    for k in K:
        u_minus[k] = model.NewIntVar(0, max_dev, f"u_minus__{k}")
        u_plus[k] = model.NewIntVar(0, max_dev, f"u_plus__{k}")
        model.Add(W_SCALE * Ck[k] + u_minus[k] - u_plus[k] == gamma[k] * Clab)
        lab_dev_terms.extend([u_minus[k], u_plus[k]])

    # Se separa el componente de campo y laboratorio para poder diagnosticar
    # específicamente la calidad de la asignación local en laboratorio.
    Z2_field = model.NewIntVar(0, len(field_dev_terms) * max_dev, "Z2_field")
    Z2_lab = model.NewIntVar(0, len(lab_dev_terms) * max_dev, "Z2_lab")
    model.Add(Z2_field == sum(field_dev_terms))
    model.Add(Z2_lab == sum(lab_dev_terms))
    Z2 = model.NewIntVar(0, (len(field_dev_terms) + len(lab_dev_terms)) * max_dev, "Z2")
    model.Add(Z2 == Z2_field + Z2_lab)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1, int(time_limit_per_stage))
    solver.parameters.num_search_workers = 8

    def is_good(status):
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def status_name(status):
        return solver.StatusName(status)

    def extract(stage_label: str, status: int) -> Dict:
        cfield_val = int(solver.Value(Cfield))
        clab_val = int(solver.Value(Clab))
        cused_val = int(solver.Value(Cused))
        total_time_by_shift = {w: int(sum(t[(f, w)] * solver.Value(Q[f]) for f in F)) for w in W}

        field_rows = []
        for f in F:
            q = int(solver.Value(Q[f]))
            day_h = int(t[(f, "Dia")] * q)
            night_h = int(t[(f, "Noche")] * q)
            cost = int(solver.Value(Cf[f]))
            field_rows.append({
                "actividad": f,
                "cantidad_Q": q,
                "tiempo_dia_h": day_h,
                "tiempo_noche_h": night_h,
                "tiempo_total_h": day_h + night_h,
                "costo_COP": cost,
                "alpha_f": alpha[f] / W_SCALE,
                "participacion_real_campo": cost / max(1, cfield_val),
            })

        lab_rows = []
        for k in K:
            cost = int(solver.Value(Ck[k]))
            target_cost = (gamma[k] / W_SCALE) * clab_val
            deviation_cost = abs(cost - target_cost)
            lab_rows.append({
                "analisis": k,
                "muestras_L": int(solver.Value(L[k])),
                "costo_COP": cost,
                "gamma_k": gamma[k] / W_SCALE,
                "participacion_objetivo_lab": gamma[k] / W_SCALE,
                "participacion_real_lab": cost / max(1, clab_val),
                "costo_objetivo_AHP_COP": int(round(target_cost)),
                "desviacion_abs_COP": int(round(deviation_cost)),
            })

        plan_rows = []
        for i in I:
            for f in F:
                for s in separations:
                    if solver.Value(y[(i, f, s)]) == 1:
                        points = int(n[(i, s)])
                        day_h = points * t[(f, "Dia")]
                        night_h = points * t[(f, "Noche")]
                        cost = points * sum(c_h[(f, w)] * t[(f, w)] for w in W)
                        plan_rows.append({
                            "circulo": i,
                            "perimetro_m": perimeter[i],
                            "actividad": f,
                            "separacion_m": s,
                            "puntos": points,
                            "tiempo_dia_h": day_h,
                            "tiempo_noche_h": night_h,
                            "tiempo_total_h": day_h + night_h,
                            "costo_COP": int(cost),
                        })

        total_samples = int(sum(int(solver.Value(Q[f])) for f in F))
        return {
            "stage": stage_label,
            "status": status_name(status),
            "objective_value": solver.ObjectiveValue(),
            "kpis": {
                "Presupuesto usado": cused_val,
                "Presupuesto disponible": budget,
                "Gasto campo": cfield_val,
                "Gasto laboratorio": clab_val,
                "Equipos dia": int(solver.Value(E["Dia"])),
                "Equipos noche": int(solver.Value(E["Noche"])),
                "Tiempo campo dia h": total_time_by_shift["Dia"],
                "Tiempo campo noche h": total_time_by_shift["Noche"],
                "Muestras campo totales": total_samples,
                "Muestras laboratorio totales": int(sum(int(solver.Value(L[k])) for k in K)),
                "Z1 aprox COP": int(solver.Value(Z1)) / W_SCALE,
                "Z2 aprox COP": int(solver.Value(Z2)) / W_SCALE,
                "Z2 campo aprox COP": int(solver.Value(Z2_field)) / W_SCALE,
                "Z2 laboratorio aprox COP": int(solver.Value(Z2_lab)) / W_SCALE,
                "Uso presupuesto %": cused_val / max(1, budget),
                "Campo real %": cfield_val / max(1, cused_val),
                "Lab real %": clab_val / max(1, cused_val),
                "Campo objetivo AHP %": beta["Campo"] / W_SCALE,
                "Lab objetivo AHP %": beta["Laboratorio"] / W_SCALE,
            },
            "field": pd.DataFrame(field_rows),
            "lab": pd.DataFrame(lab_rows),
            "plan": pd.DataFrame(plan_rows),
        }

    logs = []

    # Etapa 1: min Z1
    model.Minimize(Z1)
    status1 = solver.Solve(model)
    logs.append({"etapa": "1. Minimización de desviaciones de primer nivel", "estado": status_name(status1)})
    if not is_good(status1):
        return {"ok": False, "logs": logs, "error": "No se encontró solución factible en la etapa 1."}
    best_solution = extract("Etapa 1", status1)
    z1_value = int(solver.Value(Z1))

    # Etapa 2: fijar Z1* y min Z2
    model.Add(Z1 == z1_value)
    model.Minimize(Z2)
    status2 = solver.Solve(model)
    logs.append({"etapa": "2. Minimización de desviaciones internas con Z1=Z1*", "estado": status_name(status2)})
    if not is_good(status2):
        best_solution["warning"] = "La etapa 2 no encontró solución; se muestra etapa 1."
        best_solution["logs"] = logs
        return {"ok": True, **best_solution}
    best_solution = extract("Etapa 2", status2)
    z2_value = int(solver.Value(Z2))
    z2_field_value = int(solver.Value(Z2_field))
    z2_lab_value = int(solver.Value(Z2_lab))

    # Etapa 3: fijar Z1*, Z2* y max Cused.
    # Además, se fijan los componentes Z2_field y Z2_lab logrados en la etapa 2
    # para evitar que la maximización final del presupuesto intercambie desviación
    # entre campo y laboratorio manteniendo solo el mismo Z2 total.
    model.Add(Z2 == z2_value)
    model.Add(Z2_field == z2_field_value)
    model.Add(Z2_lab == z2_lab_value)
    model.Maximize(Cused)
    status3 = solver.Solve(model)
    logs.append({"etapa": "3. Maximización de presupuesto usado con Z1=Z1* y Z2=Z2*", "estado": status_name(status3)})
    if not is_good(status3):
        best_solution["warning"] = "La etapa 3 no encontró solución; se muestra etapa 2."
        best_solution["logs"] = logs
        return {"ok": True, **best_solution}

    final_solution = extract("Etapa 3", status3)
    final_solution["logs"] = logs
    return {"ok": True, **final_solution}


# -----------------------------
# Interfaz Streamlit
# -----------------------------

st.title("🧪 Optimización MILP-AHP para estrategias de muestreo")
st.caption("Versión estricta basada únicamente en la formulación Word: turnos Día/Noche, sin mínimos, sin reserva y sin selector de producción de muestra.")

with st.sidebar:
    st.header("1) Datos")
    st.download_button(
        "📥 Descargar plantilla Excel estricta",
        data=build_template_excel_bytes(),
        file_name="Data_template_MILP_AHP_word_strict.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        on_click="ignore",
    )
    data_mode = st.radio("Origen de datos", ["Subir Excel", "Ingresar manualmente"], index=0)
    uploaded = None
    if data_mode == "Subir Excel":
        uploaded = st.file_uploader("Carga tu Data.xlsx", type=["xlsx"])

    st.header("2) Solver")
    time_limit = st.slider("Tiempo máximo por etapa (s)", min_value=3, max_value=180, value=20, step=1)

if data_mode == "Subir Excel":
    if uploaded is None:
        st.info("Carga un Excel con la plantilla estricta o usa el modo manual.")
        st.stop()
    try:
        parsed = parse_excel(uploaded)
    except Exception as exc:
        st.error(f"No pude leer el Excel: {exc}")
        st.stop()
else:
    parsed = default_manual_data()
    st.info("Modo manual activo: edita los parámetros, círculos, separaciones, campo y laboratorio directamente en la app.")

params = parsed.params

st.subheader("Parámetros generales del modelo")
c1, c2, c3, c4 = st.columns(4)
with c1:
    budget = st.number_input("C_total: Presupuesto total COP", value=int(_safe_float(params.get("C_total"), 1)), min_value=1, step=1_000_000)
with c2:
    days = st.number_input("T_max: Días disponibles", value=int(_safe_float(params.get("T_max"), 60)), min_value=1, step=1)
with c3:
    hours = st.number_input("H_turno: Horas por turno", value=int(_safe_float(params.get("H_turno"), 8)), min_value=1, max_value=24, step=1)
with c4:
    max_teams = st.number_input("E_max: Equipos máximos por turno", value=int(_safe_float(params.get("E_max"), 2)), min_value=1, step=1)

c5, c6 = st.columns(2)
with c5:
    beta_f = st.number_input("β_F: Peso global campo", value=float(_safe_float(params.get("beta_F"), 0.642)), min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
with c6:
    beta_l = st.number_input("β_L: Peso global laboratorio", value=float(_safe_float(params.get("beta_L"), 0.358)), min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")

if abs((float(beta_f) + float(beta_l)) - 1.0) > 0.02:
    st.warning(f"β_F + β_L debería sumar 1 según el Word. Suma actual: {float(beta_f)+float(beta_l):.4f}.")

st.subheader("Conjuntos definidos por el usuario")
with st.expander("I: Círculos de hadas / perímetros", expanded=False):
    circles_df = st.data_editor(
        parsed.circles,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"Perimetro_m": st.column_config.NumberColumn("P_i: Perímetro (m)", min_value=0.01)},
    )
with st.expander("S: Separaciones permitidas", expanded=False):
    separations_df = st.data_editor(
        parsed.separations,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"separacion_m": st.column_config.NumberColumn("s: Separación (m)", min_value=1, step=1)},
    )

left, right = st.columns(2)
with left:
    st.markdown("### F: Actividades de campo")
    st.caption("No hay mínimos y no se pregunta si produce muestra: toda Q_f cuenta para la restricción Σ_k L_k ≤ Σ_f Q_f.")
    field_df = st.data_editor(
        parsed.field_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "alpha_f": st.column_config.NumberColumn("α_f", min_value=0.0, max_value=1.0, format="%.4f"),
            "t_Dia_h": st.column_config.NumberColumn("t_f,Día (h)", min_value=0, step=1),
            "t_Noche_h": st.column_config.NumberColumn("t_f,Noche (h)", min_value=0, step=1),
            "c_h_Dia_COP_h": st.column_config.NumberColumn("c_h,f,Día COP/h", min_value=0, step=10_000),
            "c_h_Noche_COP_h": st.column_config.NumberColumn("c_h,f,Noche COP/h", min_value=0, step=10_000),
        },
    )
with right:
    st.markdown("### K: Análisis de laboratorio")
    lab_df = st.data_editor(
        parsed.lab_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "gamma_k": st.column_config.NumberColumn("γ_k", min_value=0.0, max_value=1.0, format="%.4f"),
            "c_lab_COP_muestra": st.column_config.NumberColumn("c_lab,k COP/muestra", min_value=0, step=100_000),
        },
    )

# Diagnóstico de pesos locales
try:
    preview_circles, preview_sep, preview_field, preview_lab = clean_model_inputs(circles_df, separations_df, field_df, lab_df)
    st.info(
        f"Suma β = {float(beta_f)+float(beta_l):.4f}; "
        f"suma α_f = {preview_field['alpha_f'].sum():.4f}; "
        f"suma γ_k = {preview_lab['gamma_k'].sum():.4f}."
    )
except Exception as preview_exc:
    st.warning(f"Datos incompletos antes de resolver: {preview_exc}")

run = st.button("🚀 Ejecutar optimización", type="primary", use_container_width=True)

if run:
    try:
        clean_circles, sep_list, clean_field, clean_lab = clean_model_inputs(circles_df, separations_df, field_df, lab_df)
        params_used = {
            "C_total": int(budget),
            "T_max": int(days),
            "H_turno": int(hours),
            "E_max": int(max_teams),
            "beta_F": float(beta_f),
            "beta_L": float(beta_l),
            "separaciones": ", ".join(map(str, sep_list)),
        }
        with st.spinner("Resolviendo modelo lexicográfico según formulación Word..."):
            result = solve_milp_ahp_word(
                circles=clean_circles,
                separations=sep_list,
                field_df=clean_field,
                lab_df=clean_lab,
                budget_cop=int(budget),
                days_available=int(days),
                hours_per_shift=int(hours),
                max_teams=int(max_teams),
                beta_field=float(beta_f),
                beta_lab=float(beta_l),
                time_limit_per_stage=int(time_limit),
            )
    except Exception as exc:
        st.session_state.last_solution_bundle = None
        st.error(f"Error al resolver: {exc}")
        st.stop()

    if not result.get("ok"):
        st.session_state.last_solution_bundle = None
        st.error(result.get("error", "No se encontró solución."))
        st.dataframe(pd.DataFrame(result.get("logs", [])), use_container_width=True)
        st.stop()

    # Se guardan datos, solución y archivos exportables en session_state.
    # Así, si Streamlit hace rerun al descargar, no se pierde la corrida.
    excel_report = build_excel_report(result, clean_field, clean_lab, params_used)
    csv_report = result["plan"].to_csv(index=False).encode("utf-8")
    lab_by_circle, lab_long = build_lab_allocation_by_circle(result["plan"], result["lab"], clean_lab)
    st.session_state.last_solution_bundle = {
        "result": result,
        "clean_field": clean_field,
        "clean_lab": clean_lab,
        "params_used": params_used,
        "excel_report": excel_report,
        "csv_report": csv_report,
        "lab_by_circle": lab_by_circle,
        "lab_long": lab_long,
    }

bundle = st.session_state.get("last_solution_bundle")

if bundle is not None:
    result = bundle["result"]
    clean_lab = bundle["clean_lab"]
    excel_report = bundle["excel_report"]
    csv_report = bundle["csv_report"]
    lab_by_circle = bundle["lab_by_circle"]

    if not run:
        st.info("Se muestra la última solución calculada. Si cambiaste datos de entrada, ejecuta nuevamente la optimización.")

    if result.get("warning"):
        st.warning(result["warning"])

    st.success(f"Solución generada: {result['stage']} | Estado solver: {result['status']}")
    if result["status"] == "FEASIBLE":
        st.caption("FEASIBLE significa que cumple restricciones, pero el solver no demostró optimalidad dentro del tiempo definido.")

    kpis = result["kpis"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Presupuesto usado", f"${kpis['Presupuesto usado']:,.0f}", f"{kpis['Uso presupuesto %']:.1%}")
    k2.metric("Campo", f"${kpis['Gasto campo']:,.0f}", f"{kpis['Campo real %']:.1%}")
    k3.metric("Laboratorio", f"${kpis['Gasto laboratorio']:,.0f}", f"{kpis['Lab real %']:.1%}")
    k4.metric("Equipos Día/Noche", f"{kpis['Equipos dia']} / {kpis['Equipos noche']}")

    st.markdown("### Comparación AHP global vs asignación real")
    global_alloc = pd.DataFrame([
        {"bloque": "Campo", "tipo": "Objetivo AHP", "valor": kpis["Campo objetivo AHP %"]},
        {"bloque": "Campo", "tipo": "Real", "valor": kpis["Campo real %"]},
        {"bloque": "Laboratorio", "tipo": "Objetivo AHP", "valor": kpis["Lab objetivo AHP %"]},
        {"bloque": "Laboratorio", "tipo": "Real", "valor": kpis["Lab real %"]},
    ])
    fig = px.bar(global_alloc, x="bloque", y="valor", color="tipo", barmode="group", text_auto=".1%")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Participación del presupuesto", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan por círculo", "Campo", "Laboratorio", "Modelo usado", "Log solver"])
    with tab1:
        st.dataframe(result["plan"], use_container_width=True)
        c_csv, c_xlsx = st.columns(2)
        with c_csv:
            st.download_button(
                "Descargar plan CSV",
                data=csv_report,
                file_name="plan_muestreo.csv",
                mime="text/csv",
                use_container_width=True,
                on_click="ignore",
            )
        with c_xlsx:
            st.download_button(
                "Descargar reporte Excel multi-hoja",
                data=excel_report,
                file_name="reporte_muestreo_MILP_AHP_word_strict.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                on_click="ignore",
            )
    with tab2:
        st.dataframe(result["field"], use_container_width=True)
        if not result["field"].empty:
            field_plot = result["field"].melt(
                id_vars="actividad",
                value_vars=["alpha_f", "participacion_real_campo"],
                var_name="tipo",
                value_name="valor",
            )
            fig_f = px.bar(field_plot, x="actividad", y="valor", color="tipo", barmode="group", text_auto=".1%")
            fig_f.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Participación dentro de campo")
            st.plotly_chart(fig_f, use_container_width=True)
    with tab3:
        st.markdown("**Resumen global de laboratorio**")
        st.caption("La aproximación AHP local de laboratorio se evalúa sobre el gasto C_k/Clab, no sobre el número de muestras L_k. Por eso la tabla incluye costo objetivo y desviación en COP.")
        st.dataframe(result["lab"], use_container_width=True)
        st.markdown("**Asignación sugerida de laboratorio por círculo**")
        st.caption(
            "El solver decide L_k global. Esta tabla reparte cada L_k entre círculos de forma proporcional a los puntos de campo. "
            "No es una restricción adicional del modelo."
        )
        st.dataframe(lab_by_circle, use_container_width=True)
        if not result["lab"].empty:
            lab_plot = result["lab"].melt(
                id_vars="analisis",
                value_vars=["participacion_objetivo_lab", "participacion_real_lab"],
                var_name="tipo",
                value_name="valor",
            )
            fig_l = px.bar(lab_plot, x="analisis", y="valor", color="tipo", barmode="group", text_auto=".1%")
            fig_l.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Participación dentro de laboratorio")
            st.plotly_chart(fig_l, use_container_width=True)
    with tab4:
        st.dataframe(model_equations_table(), use_container_width=True)
    with tab5:
        st.dataframe(pd.DataFrame(result.get("logs", [])), use_container_width=True)
        st.json(kpis)
else:
    st.markdown(
        """
        ### Flujo sugerido
        1. Descarga la plantilla estricta o usa **Ingresar manualmente**.
        2. Define únicamente los parámetros del Word: conjuntos, costos por turno, tiempos por turno y pesos AHP.
        3. Ejecuta la optimización.
        4. Descarga el reporte Excel multi-hoja; incluye una hoja `Modelo_usado` con las ecuaciones implementadas.
        """
    )

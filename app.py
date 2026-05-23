import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model


st.set_page_config(
    page_title="MILP-AHP | Optimización de muestreo",
    page_icon="🧪",
    layout="wide",
)


# -----------------------------
# Helpers de lectura del Excel
# -----------------------------

DEFAULT_SAMPLE_PATH = "sample_data/Data.xlsx"


@dataclass
class ParsedData:
    params: Dict[str, float]
    circles: pd.DataFrame
    field_df: pd.DataFrame
    lab_df: pd.DataFrame


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clean_name_series(series: pd.Series) -> pd.Series:
    """Convierte nombres de tablas editables a texto limpio y elimina valores vacíos/NaN."""
    return (
        series
        .astype("string")
        .fillna("")
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
    )


def clean_model_inputs(
    circles: pd.DataFrame, field_df: pd.DataFrame, lab_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Limpia filas incompletas de Streamlit para evitar errores tipo `nan` en el solver."""
    circles = circles.copy()
    field_df = field_df.copy()
    lab_df = lab_df.copy()

    # Círculos válidos: ID no vacío, perímetro numérico y positivo.
    if "ID" not in circles.columns or "Perimetro_m" not in circles.columns:
        raise ValueError("La tabla de círculos debe contener las columnas ID y Perimetro_m.")
    circles["ID"] = _clean_name_series(circles["ID"])
    circles["Perimetro_m"] = pd.to_numeric(circles["Perimetro_m"], errors="coerce")
    circles = circles[(circles["ID"] != "") & circles["Perimetro_m"].notna() & (circles["Perimetro_m"] > 0)]
    circles = circles.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)

    # Actividades de campo válidas.
    required_field = ["actividad", "tiempo_h_por_punto", "peso_ahp", "minimo", "produce_muestra_lab"]
    for col in required_field:
        if col not in field_df.columns:
            field_df[col] = False if col == "produce_muestra_lab" else 0
    field_df["actividad"] = _clean_name_series(field_df["actividad"])
    field_df = field_df[field_df["actividad"] != ""].copy()
    for col in ["tiempo_h_por_punto", "peso_ahp", "minimo"]:
        field_df[col] = pd.to_numeric(field_df[col], errors="coerce").fillna(0)
    field_df["produce_muestra_lab"] = field_df["produce_muestra_lab"].fillna(False).astype(bool)
    field_df = field_df[field_df["tiempo_h_por_punto"] > 0].reset_index(drop=True)

    # Análisis de laboratorio válidos.
    required_lab = ["analisis", "costo_por_muestra", "peso_ahp", "minimo"]
    for col in required_lab:
        if col not in lab_df.columns:
            lab_df[col] = 0
    lab_df["analisis"] = _clean_name_series(lab_df["analisis"])
    lab_df = lab_df[lab_df["analisis"] != ""].copy()
    for col in ["costo_por_muestra", "peso_ahp", "minimo"]:
        lab_df[col] = pd.to_numeric(lab_df[col], errors="coerce").fillna(0)
    lab_df = lab_df[lab_df["costo_por_muestra"] >= 0].reset_index(drop=True)

    if circles.empty:
        raise ValueError("No hay círculos válidos. Revisa que cada fila tenga ID y perímetro positivo.")
    if field_df.empty:
        raise ValueError("No hay actividades de campo válidas. Revisa que cada actividad tenga nombre y tiempo > 0.")
    if lab_df.empty:
        raise ValueError("No hay análisis de laboratorio válidos. Revisa que cada análisis tenga nombre.")

    return circles, field_df, lab_df


def parse_excel(file_obj) -> ParsedData:
    """Lee Data.xlsx y lo transforma en tablas útiles para el modelo."""
    master = pd.read_excel(file_obj, sheet_name="Master_Datos")
    circles = pd.read_excel(file_obj, sheet_name="Sheet2")

    # Normalización mínima de columnas
    master = master.iloc[:, :3].copy()
    master.columns = ["Parametro", "Unidad", "Valor"]
    master = master.dropna(subset=["Parametro"])
    master["Parametro"] = master["Parametro"].astype(str).str.strip()
    params = dict(zip(master["Parametro"], master["Valor"]))

    circles = circles.iloc[:, :2].copy()
    circles.columns = ["ID", "Perimetro_m"]
    circles = circles.dropna(subset=["ID", "Perimetro_m"])
    circles["ID"] = circles["ID"].astype(str).str.strip()
    circles["Perimetro_m"] = pd.to_numeric(circles["Perimetro_m"], errors="coerce")
    circles = circles.dropna(subset=["Perimetro_m"]).reset_index(drop=True)

    # Actividades de campo según el Excel actual
    field_df = pd.DataFrame(
        [
            {
                "actividad": "Muestreo puntual estándar",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo requerido para tomar un muestreo puntual estándar"), 1),
                "peso_ahp": _safe_float(params.get("Calidad de un punto de muestreo estándar"), 0.0),
                "minimo": _safe_float(params.get("Candidatos de muestreo - Teledetección"), 0.0),
                "produce_muestra_lab": True,
            },
            {
                "actividad": "Muestreo multinivel",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo requerido para hacer un muestreo multinivel"), 3),
                "peso_ahp": _safe_float(params.get("Calidad de un punto multinivel"), 0.0),
                "minimo": _safe_float(params.get("Puntos multinivel mínimos por círculo de hadas"), 0.0) * max(len(circles), 1),
                "produce_muestra_lab": True,
            },
            {
                "actividad": "Medición 24 horas",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo que consume una medición de 24 horas"), 25),
                "peso_ahp": _safe_float(params.get("Calidad de una medición de 24 horas"), 0.0),
                "minimo": _safe_float(params.get("Mediciones de 24 horas mínimas en toda la estrategia"), 0.0),
                "produce_muestra_lab": False,
            },
            {
                "actividad": "Medición extendida",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo que consume una medición extendida (5 días)"), 121),
                "peso_ahp": _safe_float(params.get("Calidad de una medición extendida"), 0.0),
                "minimo": _safe_float(params.get("Mediciones extendidas mínimas en toda la estrategia"), 0.0),
                "produce_muestra_lab": False,
            },
        ]
    )

    isotopic_cost = _safe_float(params.get("Análisis de Abundancia Isotópica"), 0.0)
    lab_df = pd.DataFrame(
        [
            {
                "analisis": "Cromatografía de Gas",
                "costo_por_muestra": _safe_float(params.get("Análisis de Cromatrografía de Gas"), 0.0),
                "peso_ahp": _safe_float(params.get("Calidad de cromatografia"), 0.0),
                "minimo": _safe_float(params.get("Muestras mínimas para cromatografia"), 0.0),
            },
            {
                "analisis": "Isotopía de Deuterio",
                "costo_por_muestra": isotopic_cost,
                "peso_ahp": _safe_float(params.get("Calidad de isotopia de Deuterio"), 0.0),
                "minimo": _safe_float(params.get("Muestras mínimas para isotopia de Deuterio"), 0.0),
            },
            {
                "analisis": "Isotopía de Helio",
                "costo_por_muestra": isotopic_cost,
                "peso_ahp": _safe_float(params.get("Calidad de isotopia de Helio"), 0.0),
                "minimo": _safe_float(params.get("Muestras mínimas para isotopia de Helio"), 0.0),
            },
            {
                "analisis": "Caracterización mineralógica",
                "costo_por_muestra": _safe_float(params.get("Caracterización Mineralógica"), 0.0),
                "peso_ahp": _safe_float(params.get("Calidad de caracterizacion mineralogica"), 0.0),
                "minimo": _safe_float(params.get("Muestras mínimas para caracterizacion mineralogica"), 0.0),
            },
            {
                "analisis": "Biogeoquímica",
                "costo_por_muestra": _safe_float(params.get("Análisis Biogeoquímico"), 0.0),
                "peso_ahp": _safe_float(params.get("Calidad de biogeoquimica"), 0.0),
                "minimo": _safe_float(params.get("Muestras mínimas para biogeoquimica"), 0.0),
            },
        ]
    )

    return ParsedData(params=params, circles=circles, field_df=field_df, lab_df=lab_df)


def parse_separations(text: str) -> List[int]:
    vals = []
    for raw in text.replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        val = int(round(float(raw)))
        if val > 0:
            vals.append(val)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("Debes introducir al menos una separación positiva.")
    return vals


def compute_n_points(perimeter: float, separation: int, mode: str) -> int:
    if separation <= 0:
        raise ValueError("Todas las separaciones deben ser mayores que cero.")
    if not math.isfinite(float(perimeter)) or perimeter <= 0:
        raise ValueError(f"Perímetro inválido: {perimeter}. Revisa la tabla de círculos.")
    ratio = perimeter / separation
    if mode == "ceil":
        val = math.ceil(ratio)
    elif mode == "round":
        val = round(ratio)
    else:
        val = math.floor(ratio)
    return max(1, int(val))


# -----------------------------
# Solver CP-SAT
# -----------------------------


def solve_milp_ahp(
    circles: pd.DataFrame,
    field_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    separations: List[int],
    budget_cop: int,
    days_available: int,
    hours_per_day: int,
    max_teams: int,
    logistic_daily_cost: int,
    n_mode: str,
    enforce_minimums: bool,
    lab_cap_enabled: bool,
    allow_multiple_activities: bool,
    time_limit_per_stage: int,
    relative_tolerance: float,
) -> Dict:
    """Resuelve el modelo lexicográfico con OR-Tools CP-SAT.

    Los costos se redondean a COP enteros para mejorar estabilidad y velocidad.
    Los pesos AHP se escalan a enteros con W_SCALE.
    """
    circles, field_df, lab_df = clean_model_inputs(circles, field_df, lab_df)

    # Limpieza numérica adicional: evita NaN cuando Streamlit deja celdas vacías.
    for col in ["tiempo_h_por_punto", "peso_ahp", "minimo"]:
        field_df[col] = pd.to_numeric(field_df[col], errors="coerce").fillna(0)
    for col in ["costo_por_muestra", "peso_ahp", "minimo"]:
        lab_df[col] = pd.to_numeric(lab_df[col], errors="coerce").fillna(0)

    F = field_df["actividad"].tolist()
    K = lab_df["analisis"].tolist()
    I = circles["ID"].tolist()
    perimeter = dict(zip(I, circles["Perimetro_m"].astype(float)))

    # Pesos AHP globales y locales
    field_weight_sum = float(field_df["peso_ahp"].sum())
    lab_weight_sum = float(lab_df["peso_ahp"].sum())
    total_weight = field_weight_sum + lab_weight_sum
    if total_weight <= 0:
        raise ValueError("La suma de pesos AHP debe ser mayor que cero.")
    if field_weight_sum <= 0 or lab_weight_sum <= 0:
        raise ValueError("Debe haber pesos AHP positivos tanto en campo como en laboratorio.")

    W_SCALE = 10_000
    beta_f = int(round((field_weight_sum / total_weight) * W_SCALE))
    beta_l = int(round((lab_weight_sum / total_weight) * W_SCALE))
    alpha = {
        row["actividad"]: int(round((row["peso_ahp"] / field_weight_sum) * W_SCALE))
        for _, row in field_df.iterrows()
    }
    gamma = {
        row["analisis"]: int(round((row["peso_ahp"] / lab_weight_sum) * W_SCALE))
        for _, row in lab_df.iterrows()
    }

    # Parámetros enteros
    budget = int(round(budget_cop))
    hourly_logistic_cost = logistic_daily_cost / max(hours_per_day, 1)
    t = dict(zip(F, field_df["tiempo_h_por_punto"].round().astype(int)))
    field_cost = {f: int(round(hourly_logistic_cost * t[f])) for f in F}
    lab_cost = dict(zip(K, lab_df["costo_por_muestra"].round().astype(int)))
    produce_lab_sample = dict(zip(F, field_df["produce_muestra_lab"].astype(bool)))

    n = {(i, s): compute_n_points(perimeter[i], s, n_mode) for i in I for s in separations}

    model = cp_model.CpModel()

    # Variables de decisión
    y = {(i, f, s): model.NewBoolVar(f"y__{i}__{f}__{s}") for i in I for f in F for s in separations}
    max_q = sum(max(n[(i, s)] for s in separations) for i in I)
    Q = {f: model.NewIntVar(0, max_q, f"Q__{f}") for f in F}
    Cf = {f: model.NewIntVar(0, budget, f"Cf__{f}") for f in F}

    min_lab_cost = max(1, min([v for v in lab_cost.values() if v > 0] or [1]))
    lab_upper_bound = int(budget / min_lab_cost) + max_q + 100
    L = {k: model.NewIntVar(0, lab_upper_bound, f"L__{k}") for k in K}
    Ck = {k: model.NewIntVar(0, budget, f"Ck__{k}") for k in K}

    Cfield = model.NewIntVar(0, budget, "Cfield")
    Clab = model.NewIntVar(0, budget, "Clab")
    Cused = model.NewIntVar(0, budget, "Cused")
    E = model.NewIntVar(1, max_teams, "E")

    # Geometría y activación
    for i in I:
        if allow_multiple_activities:
            for f in F:
                model.Add(sum(y[(i, f, s)] for s in separations) <= 1)
            model.Add(sum(y[(i, f, s)] for f in F for s in separations) >= 1)
        else:
            model.Add(sum(y[(i, f, s)] for f in F for s in separations) == 1)

    # Cantidades y costos de campo
    for f in F:
        model.Add(Q[f] == sum(n[(i, s)] * y[(i, f, s)] for i in I for s in separations))
        model.Add(Cf[f] == field_cost[f] * Q[f])

    # Cantidades y costos de laboratorio
    for k in K:
        model.Add(Ck[k] == lab_cost[k] * L[k])

    model.Add(Cfield == sum(Cf[f] for f in F))
    model.Add(Clab == sum(Ck[k] for k in K))
    model.Add(Cused == Cfield + Clab)
    model.Add(Cused <= budget)

    # Capacidad operativa
    model.Add(sum(t[f] * Q[f] for f in F) <= E * days_available * hours_per_day)

    # Mínimos opcionales
    if enforce_minimums:
        for _, row in field_df.iterrows():
            if row["minimo"] > 0:
                model.Add(Q[row["actividad"]] >= int(round(row["minimo"])))
        for _, row in lab_df.iterrows():
            if row["minimo"] > 0:
                model.Add(L[row["analisis"]] >= int(round(row["minimo"])))

    # Capacidad de laboratorio opcional: cada análisis no puede superar muestras físicas producidas
    if lab_cap_enabled:
        sample_capacity = sum(Q[f] for f in F if produce_lab_sample.get(f, False))
        for k in K:
            model.Add(L[k] <= sample_capacity)

    # Desviaciones AHP como valores absolutos enteros escalados
    max_dev = budget * W_SCALE
    dev_field_global = model.NewIntVar(0, max_dev, "dev_field_global")
    dev_lab_global = model.NewIntVar(0, max_dev, "dev_lab_global")
    model.AddAbsEquality(dev_field_global, W_SCALE * Cfield - beta_f * Cused)
    model.AddAbsEquality(dev_lab_global, W_SCALE * Clab - beta_l * Cused)
    Z1 = model.NewIntVar(0, 2 * max_dev, "Z1")
    model.Add(Z1 == dev_field_global + dev_lab_global)

    local_devs = []
    for f in F:
        d = model.NewIntVar(0, max_dev, f"dev_field__{f}")
        model.AddAbsEquality(d, W_SCALE * Cf[f] - alpha[f] * Cfield)
        local_devs.append(d)
    for k in K:
        d = model.NewIntVar(0, max_dev, f"dev_lab__{k}")
        model.AddAbsEquality(d, W_SCALE * Ck[k] - gamma[k] * Clab)
        local_devs.append(d)
    Z2 = model.NewIntVar(0, len(local_devs) * max_dev, "Z2")
    model.Add(Z2 == sum(local_devs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1, int(time_limit_per_stage))
    solver.parameters.num_search_workers = 8

    def is_good(status):
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def status_name(status):
        return solver.StatusName(status)

    def extract(stage_label: str, status: int) -> Dict:
        field_rows = []
        total_time = 0
        for f in F:
            q = int(solver.Value(Q[f]))
            cost = int(solver.Value(Cf[f]))
            time_h = int(t[f] * q)
            total_time += time_h
            field_rows.append(
                {
                    "actividad": f,
                    "cantidad_Q": q,
                    "tiempo_h": time_h,
                    "costo_COP": cost,
                    "peso_ahp_local": alpha[f] / W_SCALE,
                    "participacion_real_campo": cost / max(1, solver.Value(Cfield)),
                }
            )

        lab_rows = []
        for k in K:
            cost = int(solver.Value(Ck[k]))
            lab_rows.append(
                {
                    "analisis": k,
                    "muestras_L": int(solver.Value(L[k])),
                    "costo_COP": cost,
                    "peso_ahp_local": gamma[k] / W_SCALE,
                    "participacion_real_lab": cost / max(1, solver.Value(Clab)),
                }
            )

        plan_rows = []
        for i in I:
            for f in F:
                for s in separations:
                    if solver.Value(y[(i, f, s)]) == 1:
                        points = n[(i, s)]
                        plan_rows.append(
                            {
                                "circulo": i,
                                "perimetro_m": perimeter[i],
                                "actividad": f,
                                "separacion_m": s,
                                "puntos": points,
                                "tiempo_h": points * t[f],
                                "costo_COP": points * field_cost[f],
                            }
                        )

        cfield_val = int(solver.Value(Cfield))
        clab_val = int(solver.Value(Clab))
        cused_val = int(solver.Value(Cused))
        return {
            "stage": stage_label,
            "status": status_name(status),
            "objective_value": solver.ObjectiveValue(),
            "kpis": {
                "Presupuesto usado": cused_val,
                "Presupuesto disponible": budget,
                "Gasto campo": cfield_val,
                "Gasto laboratorio": clab_val,
                "Equipos": int(solver.Value(E)),
                "Tiempo campo h": total_time,
                "Z1 aprox COP": solver.Value(Z1) / W_SCALE,
                "Z2 aprox COP": solver.Value(Z2) / W_SCALE,
                "Uso presupuesto %": cused_val / max(1, budget),
                "Campo real %": cfield_val / max(1, cused_val),
                "Lab real %": clab_val / max(1, cused_val),
                "Campo objetivo AHP %": beta_f / W_SCALE,
                "Lab objetivo AHP %": beta_l / W_SCALE,
            },
            "field": pd.DataFrame(field_rows),
            "lab": pd.DataFrame(lab_rows),
            "plan": pd.DataFrame(plan_rows),
        }

    logs = []

    # Etapa 1
    model.Minimize(Z1)
    status1 = solver.Solve(model)
    logs.append({"etapa": "1. Equilibrio Campo/Lab", "estado": status_name(status1)})
    if not is_good(status1):
        return {"ok": False, "logs": logs, "error": "No se encontró solución factible en la etapa 1."}
    best_solution = extract("Etapa 1", status1)
    z1_value = int(solver.Value(Z1))

    # Etapa 2
    tolerance = int(max(W_SCALE, budget * W_SCALE * relative_tolerance))
    model.Add(Z1 <= z1_value + tolerance)
    model.Minimize(Z2)
    status2 = solver.Solve(model)
    logs.append({"etapa": "2. Equilibrio interno", "estado": status_name(status2)})
    if not is_good(status2):
        best_solution["warning"] = "La etapa 2 no encontró solución dentro del límite; se muestra la mejor solución de etapa 1."
        best_solution["logs"] = logs
        return {"ok": True, **best_solution}
    best_solution = extract("Etapa 2", status2)
    z2_value = int(solver.Value(Z2))

    # Etapa 3
    model.Add(Z2 <= z2_value + tolerance)
    model.Maximize(Cused)
    status3 = solver.Solve(model)
    logs.append({"etapa": "3. Maximización de presupuesto usado", "estado": status_name(status3)})
    if not is_good(status3):
        best_solution["warning"] = "La etapa 3 no encontró solución dentro del límite; se muestra la mejor solución de etapa 2."
        best_solution["logs"] = logs
        return {"ok": True, **best_solution}

    final_solution = extract("Etapa 3", status3)
    final_solution["logs"] = logs
    return {"ok": True, **final_solution}


# -----------------------------
# UI
# -----------------------------

st.title("🧪 Optimización MILP-AHP para estrategias de muestreo")
st.caption("Asignación de recursos entre campo y laboratorio usando prioridades AHP, restricciones geométricas, presupuesto y capacidad operativa.")

with st.sidebar:
    st.header("1) Datos")
    uploaded = st.file_uploader("Carga tu Data.xlsx", type=["xlsx"])
    use_example = st.checkbox("Usar archivo de ejemplo incluido", value=uploaded is None)

    st.header("2) Solver")
    separations_text = st.text_input("Separaciones permitidas en metros", value="10, 20, 50")
    n_mode_label = st.selectbox(
        "Cálculo de puntos por perímetro",
        options=["floor", "ceil", "round"],
        index=0,
        help="floor = cuántos puntos caben sin exceder el perímetro; ceil = cubre todo el perímetro.",
    )
    allow_multiple = st.checkbox(
        "Permitir varias actividades por círculo (formulación Word)",
        value=True,
        help="Más fiel al modelo original, pero puede tardar más. Si el solver no encuentra solución rápido, desactívalo.",
    )
    enforce_minimums = st.checkbox(
        "Activar mínimos del Excel",
        value=False,
        help="El documento lógico indica que no se imponen mínimos por actividad; por eso queda desactivado por defecto.",
    )
    lab_cap_enabled = st.checkbox(
        "Limitar muestras de laboratorio por muestras de campo producidas",
        value=True,
    )
    time_limit = st.slider("Tiempo máximo por etapa (s)", min_value=3, max_value=120, value=15, step=1)
    relative_tolerance = st.select_slider(
        "Tolerancia lexicográfica relativa",
        options=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        value=1e-6,
        format_func=lambda x: f"{x:g}",
    )

# Carga inicial
try:
    if uploaded is not None:
        parsed = parse_excel(uploaded)
    elif use_example:
        parsed = parse_excel(DEFAULT_SAMPLE_PATH)
    else:
        st.info("Carga un archivo Excel para comenzar.")
        st.stop()
except Exception as exc:
    st.error(f"No pude leer el Excel: {exc}")
    st.stop()

params = parsed.params

st.subheader("Parámetros generales")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    budget = st.number_input("Presupuesto total COP", value=int(_safe_float(params.get("Presupuesto Total Disponible"), 0)), min_value=1, step=1_000_000)
with c2:
    days = st.number_input("Días disponibles", value=int(_safe_float(params.get("Tiempo total disponible para el muestreo"), 60)), min_value=1, step=1)
with c3:
    hours = st.number_input("Horas/día", value=int(_safe_float(params.get("Horas efectivas de trabajo del equipo por día"), 8)), min_value=1, max_value=24, step=1)
with c4:
    max_teams = st.number_input("Equipos/sensores máx.", value=int(_safe_float(params.get("Cantidad de equipos/sensores disponibles"), 2)), min_value=1, step=1)
with c5:
    logistic_daily = st.number_input("Costo logístico diario COP", value=int(_safe_float(params.get("Logística de Muestreo (Honorarios, Viáticos y Transporte)"), 0)), min_value=0, step=100_000)

st.subheader("Círculos de hadas / perímetros")
with st.expander("Ver y editar círculos", expanded=False):
    circles_df = st.data_editor(
        parsed.circles,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"Perimetro_m": st.column_config.NumberColumn("Perímetro (m)", min_value=0.01)},
    )

st.subheader("Actividades y análisis")
left, right = st.columns(2)
with left:
    st.markdown("**Campo**")
    field_df = st.data_editor(
        parsed.field_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "tiempo_h_por_punto": st.column_config.NumberColumn("Tiempo h/punto", min_value=0),
            "peso_ahp": st.column_config.NumberColumn("Peso AHP", min_value=0.0, format="%.4f"),
            "minimo": st.column_config.NumberColumn("Mínimo", min_value=0),
            "produce_muestra_lab": st.column_config.CheckboxColumn("Produce muestra lab"),
        },
    )
with right:
    st.markdown("**Laboratorio**")
    lab_df = st.data_editor(
        parsed.lab_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "costo_por_muestra": st.column_config.NumberColumn("Costo/muestra COP", min_value=0, step=100_000),
            "peso_ahp": st.column_config.NumberColumn("Peso AHP", min_value=0.0, format="%.4f"),
            "minimo": st.column_config.NumberColumn("Mínimo", min_value=0),
        },
    )

# Diagnóstico AHP previo
try:
    preview_circles, preview_field_df, preview_lab_df = clean_model_inputs(circles_df, field_df, lab_df)
except Exception as preview_exc:
    st.warning(f"Hay filas incompletas o inválidas antes de resolver: {preview_exc}")
    preview_field_df = field_df.copy()
    preview_lab_df = lab_df.copy()

field_weight = pd.to_numeric(preview_field_df["peso_ahp"], errors="coerce").fillna(0).sum()
lab_weight = pd.to_numeric(preview_lab_df["peso_ahp"], errors="coerce").fillna(0).sum()
total_weight = field_weight + lab_weight
if total_weight > 0:
    st.info(
        f"Objetivo AHP global: Campo = {field_weight / total_weight:.2%}, "
        f"Laboratorio = {lab_weight / total_weight:.2%}."
    )

run = st.button("🚀 Ejecutar optimización", type="primary", use_container_width=True)

if run:
    try:
        separations = parse_separations(separations_text)
        with st.spinner("Resolviendo modelo lexicográfico..."):
            clean_circles_df, clean_field_df, clean_lab_df = clean_model_inputs(circles_df, field_df, lab_df)
            result = solve_milp_ahp(
                circles=clean_circles_df,
                field_df=clean_field_df,
                lab_df=clean_lab_df,
                separations=separations,
                budget_cop=int(budget),
                days_available=int(days),
                hours_per_day=int(hours),
                max_teams=int(max_teams),
                logistic_daily_cost=int(logistic_daily),
                n_mode=n_mode_label,
                enforce_minimums=enforce_minimums,
                lab_cap_enabled=lab_cap_enabled,
                allow_multiple_activities=allow_multiple,
                time_limit_per_stage=int(time_limit),
                relative_tolerance=float(relative_tolerance),
            )
    except Exception as exc:
        st.error(f"Error al resolver: {exc}")
        st.stop()

    if not result.get("ok"):
        st.error(result.get("error", "No se encontró solución."))
        st.dataframe(pd.DataFrame(result.get("logs", [])), use_container_width=True)
        st.stop()

    if result.get("warning"):
        st.warning(result["warning"])

    st.success(f"Solución generada: {result['stage']} | Estado solver: {result['status']}")
    if result["status"] == "FEASIBLE":
        st.caption("FEASIBLE significa que la solución cumple restricciones, pero el solver no demostró optimalidad dentro del tiempo definido.")

    kpis = result["kpis"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Presupuesto usado", f"${kpis['Presupuesto usado']:,.0f}", f"{kpis['Uso presupuesto %']:.1%}")
    k2.metric("Campo", f"${kpis['Gasto campo']:,.0f}", f"{kpis['Campo real %']:.1%}")
    k3.metric("Laboratorio", f"${kpis['Gasto laboratorio']:,.0f}", f"{kpis['Lab real %']:.1%}")
    k4.metric("Tiempo / equipos", f"{kpis['Tiempo campo h']:,.0f} h", f"{kpis['Equipos']} equipo(s)")

    st.markdown("### Comparación AHP vs asignación real")
    global_alloc = pd.DataFrame(
        [
            {"bloque": "Campo", "tipo": "Objetivo AHP", "valor": kpis["Campo objetivo AHP %"]},
            {"bloque": "Campo", "tipo": "Real", "valor": kpis["Campo real %"]},
            {"bloque": "Laboratorio", "tipo": "Objetivo AHP", "valor": kpis["Lab objetivo AHP %"]},
            {"bloque": "Laboratorio", "tipo": "Real", "valor": kpis["Lab real %"]},
        ]
    )
    fig = px.bar(global_alloc, x="bloque", y="valor", color="tipo", barmode="group", text_auto=".1%")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Participación del presupuesto", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Plan por círculo", "Campo", "Laboratorio", "Log solver"])
    with tab1:
        st.dataframe(result["plan"], use_container_width=True)
        st.download_button(
            "Descargar plan CSV",
            data=result["plan"].to_csv(index=False).encode("utf-8"),
            file_name="plan_muestreo.csv",
            mime="text/csv",
        )
    with tab2:
        st.dataframe(result["field"], use_container_width=True)
        if not result["field"].empty:
            field_plot = result["field"].melt(
                id_vars="actividad",
                value_vars=["peso_ahp_local", "participacion_real_campo"],
                var_name="tipo",
                value_name="valor",
            )
            fig_f = px.bar(field_plot, x="actividad", y="valor", color="tipo", barmode="group", text_auto=".1%")
            fig_f.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Participación dentro de campo")
            st.plotly_chart(fig_f, use_container_width=True)
    with tab3:
        st.dataframe(result["lab"], use_container_width=True)
        if not result["lab"].empty:
            lab_plot = result["lab"].melt(
                id_vars="analisis",
                value_vars=["peso_ahp_local", "participacion_real_lab"],
                var_name="tipo",
                value_name="valor",
            )
            fig_l = px.bar(lab_plot, x="analisis", y="valor", color="tipo", barmode="group", text_auto=".1%")
            fig_l.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Participación dentro de laboratorio")
            st.plotly_chart(fig_l, use_container_width=True)
    with tab4:
        st.dataframe(pd.DataFrame(result.get("logs", [])), use_container_width=True)
        st.json(kpis)
else:
    st.markdown(
        """
        ### Flujo sugerido
        1. Revisa o edita los parámetros generales.
        2. Ajusta actividades de campo, análisis de laboratorio y pesos AHP.
        3. Define las separaciones permitidas.
        4. Ejecuta la optimización.

        **Nota:** si el solver tarda o devuelve solo `FEASIBLE`, prueba con menos separaciones o desactiva
        “Permitir varias actividades por círculo” para usar una versión más rápida del problema.
        """
    )

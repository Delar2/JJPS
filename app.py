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



@dataclass
class ParsedData:
    params: Dict[str, float]
    circles: pd.DataFrame
    field_df: pd.DataFrame
    lab_df: pd.DataFrame



GENERAL_PARAM_ROWS = [
    {"Parametro": "Tiempo total disponible para el muestreo", "Unidad": "Días", "Valor": 60},
    {"Parametro": "Horas efectivas de trabajo del equipo por día", "Unidad": "Horas", "Valor": 8},
    {"Parametro": "Presupuesto Total Disponible", "Unidad": "COP", "Valor": 446852500},
    {"Parametro": "Logística de Muestreo (Honorarios, Viáticos y Transporte)", "Unidad": "COP/día", "Valor": 1513000},
    {"Parametro": "Cantidad de equipos/sensores disponibles", "Unidad": "Unidad", "Valor": 2},
]

DEFAULT_FIELD_ROWS = [
    {"actividad": "Muestreo puntual estándar", "tiempo_h_por_punto": 1, "peso_ahp": 0.2417, "produce_muestra_lab": True},
    {"actividad": "Muestreo multinivel", "tiempo_h_por_punto": 3, "peso_ahp": 0.2029, "produce_muestra_lab": True},
    {"actividad": "Medición 24 horas", "tiempo_h_por_punto": 25, "peso_ahp": 0.1459, "produce_muestra_lab": False},
    {"actividad": "Medición extendida", "tiempo_h_por_punto": 121, "peso_ahp": 0.0523, "produce_muestra_lab": False},
]

DEFAULT_LAB_ROWS = [
    {"analisis": "Cromatografía de Gas", "costo_por_muestra": 1190000, "peso_ahp": 0.1511},
    {"analisis": "Isotopía de Deuterio", "costo_por_muestra": 6538812, "peso_ahp": 0.0897},
    {"analisis": "Isotopía de Helio", "costo_por_muestra": 6538812, "peso_ahp": 0.0438},
    {"analisis": "Caracterización mineralógica", "costo_por_muestra": 1953385, "peso_ahp": 0.0399},
    {"analisis": "Biogeoquímica", "costo_por_muestra": 3015249, "peso_ahp": 0.0328},
]


def default_manual_data() -> ParsedData:
    """Datos iniciales para usar la app sin Excel. El usuario puede editar todo a mano."""
    params_df = pd.DataFrame(GENERAL_PARAM_ROWS)
    params = dict(zip(params_df["Parametro"], params_df["Valor"]))
    circles = pd.DataFrame([
        {"ID": "C1", "Perimetro_m": 100.0},
        {"ID": "C2", "Perimetro_m": 150.0},
        {"ID": "C3", "Perimetro_m": 250.0},
    ])
    return ParsedData(
        params=params,
        circles=circles,
        field_df=pd.DataFrame(DEFAULT_FIELD_ROWS),
        lab_df=pd.DataFrame(DEFAULT_LAB_ROWS),
    )


def build_template_excel_bytes() -> bytes:
    """Plantilla descargable con el formato recomendado para la app."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(GENERAL_PARAM_ROWS).to_excel(writer, sheet_name="Parametros", index=False)
        pd.DataFrame([
            {"ID": "C1", "Perimetro_m": 113.0},
            {"ID": "C2", "Perimetro_m": 164.0},
            {"ID": "C3", "Perimetro_m": 702.0},
        ]).to_excel(writer, sheet_name="Circulos", index=False)
        pd.DataFrame(DEFAULT_FIELD_ROWS).to_excel(writer, sheet_name="Campo", index=False)
        pd.DataFrame(DEFAULT_LAB_ROWS).to_excel(writer, sheet_name="Laboratorio", index=False)
        notes = pd.DataFrame([
            {"Hoja": "Parametros", "Descripcion": "Parámetros generales del modelo. Debe conservar columnas: Parametro, Unidad, Valor."},
            {"Hoja": "Circulos", "Descripcion": "Lista de círculos de hadas. Debe conservar columnas: ID, Perimetro_m."},
            {"Hoja": "Campo", "Descripcion": "Actividades de campo. produce_muestra_lab=True si cada punto genera una muestra física para laboratorio."},
            {"Hoja": "Laboratorio", "Descripcion": "Tipos de análisis de laboratorio, costo por muestra y peso AHP."},
            {"Hoja": "Importante", "Descripcion": "La restricción actual usa conservación total: suma de muestras de laboratorio <= muestras físicas recolectadas en campo."},
        ])
        notes.to_excel(writer, sheet_name="Notas_formato", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                max_len = 10
                for cell in column_cells[:100]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[column_cells[0].column_letter].width = min(max_len + 2, 55)
    return output.getvalue()


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
    required_field = ["actividad", "tiempo_h_por_punto", "peso_ahp", "produce_muestra_lab"]
    for col in required_field:
        if col not in field_df.columns:
            field_df[col] = False if col == "produce_muestra_lab" else 0
    field_df["actividad"] = _clean_name_series(field_df["actividad"])
    field_df = field_df[field_df["actividad"] != ""].copy()
    for col in ["tiempo_h_por_punto", "peso_ahp"]:
        field_df[col] = pd.to_numeric(field_df[col], errors="coerce").fillna(0)
    field_df["produce_muestra_lab"] = field_df["produce_muestra_lab"].fillna(False).astype(bool)
    field_df = field_df[field_df["tiempo_h_por_punto"] > 0].reset_index(drop=True)

    # Análisis de laboratorio válidos.
    required_lab = ["analisis", "costo_por_muestra", "peso_ahp"]
    for col in required_lab:
        if col not in lab_df.columns:
            lab_df[col] = 0
    lab_df["analisis"] = _clean_name_series(lab_df["analisis"])
    lab_df = lab_df[lab_df["analisis"] != ""].copy()
    for col in ["costo_por_muestra", "peso_ahp"]:
        lab_df[col] = pd.to_numeric(lab_df[col], errors="coerce").fillna(0)
    lab_df = lab_df[lab_df["costo_por_muestra"] >= 0].reset_index(drop=True)

    if circles.empty:
        raise ValueError("No hay círculos válidos. Revisa que cada fila tenga ID y perímetro positivo.")
    if field_df.empty:
        raise ValueError("No hay actividades de campo válidas. Revisa que cada actividad tenga nombre y tiempo > 0.")
    if lab_df.empty:
        raise ValueError("No hay análisis de laboratorio válidos. Revisa que cada análisis tenga nombre.")

    return circles, field_df, lab_df



def _read_params_table(df: pd.DataFrame) -> Dict[str, float]:
    df = df.iloc[:, :3].copy()
    df.columns = ["Parametro", "Unidad", "Valor"]
    df = df.dropna(subset=["Parametro"])
    df["Parametro"] = df["Parametro"].astype(str).str.strip()
    return dict(zip(df["Parametro"], df["Valor"]))


def _parse_new_template(xls: pd.ExcelFile) -> ParsedData:
    sheet_map = {s.lower().strip(): s for s in xls.sheet_names}
    params_df = xls.parse(sheet_map["parametros"])
    params = _read_params_table(params_df)

    circles = xls.parse(sheet_map["circulos"])
    if len(circles.columns) < 2:
        raise ValueError("La hoja Circulos debe tener al menos dos columnas: ID y Perimetro_m.")
    circles = circles.iloc[:, :2].copy()
    circles.columns = ["ID", "Perimetro_m"]
    circles["ID"] = _clean_name_series(circles["ID"])
    circles["Perimetro_m"] = pd.to_numeric(circles["Perimetro_m"], errors="coerce")
    circles = circles[(circles["ID"] != "") & circles["Perimetro_m"].notna()].reset_index(drop=True)

    field_df = xls.parse(sheet_map["campo"])
    lab_df = xls.parse(sheet_map["laboratorio"])

    # Normalizar columnas esperadas del formato nuevo.
    # v4 ya no usa mínimos; por compatibilidad, si el Excel antiguo trae una columna minimo, se ignora.
    field_expected = ["actividad", "tiempo_h_por_punto", "peso_ahp", "produce_muestra_lab"]
    field_legacy = ["actividad", "tiempo_h_por_punto", "peso_ahp", "minimo", "produce_muestra_lab"]
    lab_expected = ["analisis", "costo_por_muestra", "peso_ahp"]
    lab_legacy = ["analisis", "costo_por_muestra", "peso_ahp", "minimo"]

    if len(field_df.columns) >= len(field_legacy):
        field_df = field_df.iloc[:, :len(field_legacy)].copy()
        field_df.columns = field_legacy
        field_df = field_df[field_expected]
    elif len(field_df.columns) >= len(field_expected):
        field_df = field_df.iloc[:, :len(field_expected)].copy()
        field_df.columns = field_expected
    else:
        raise ValueError(f"La hoja Campo debe tener columnas: {', '.join(field_expected)}")

    if len(lab_df.columns) >= len(lab_legacy):
        lab_df = lab_df.iloc[:, :len(lab_legacy)].copy()
        lab_df.columns = lab_legacy
        lab_df = lab_df[lab_expected]
    elif len(lab_df.columns) >= len(lab_expected):
        lab_df = lab_df.iloc[:, :len(lab_expected)].copy()
        lab_df.columns = lab_expected
    else:
        raise ValueError(f"La hoja Laboratorio debe tener columnas: {', '.join(lab_expected)}")

    return ParsedData(params=params, circles=circles, field_df=field_df, lab_df=lab_df)


def _parse_legacy_excel(xls: pd.ExcelFile) -> ParsedData:
    """Lee el Data.xlsx original con hojas Master_Datos y Sheet2."""
    master = xls.parse("Master_Datos")
    circles = xls.parse("Sheet2")

    # Normalización mínima de columnas
    params = _read_params_table(master)

    circles = circles.iloc[:, :2].copy()
    circles.columns = ["ID", "Perimetro_m"]
    circles = circles.dropna(subset=["ID", "Perimetro_m"])
    circles["ID"] = circles["ID"].astype(str).str.strip()
    circles["Perimetro_m"] = pd.to_numeric(circles["Perimetro_m"], errors="coerce")
    circles = circles.dropna(subset=["Perimetro_m"]).reset_index(drop=True)

    # Actividades de campo según el Excel original
    field_df = pd.DataFrame(
        [
            {
                "actividad": "Muestreo puntual estándar",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo requerido para tomar un muestreo puntual estándar"), 1),
                "peso_ahp": _safe_float(params.get("Calidad de un punto de muestreo estándar"), 0.0),
                "produce_muestra_lab": True,
            },
            {
                "actividad": "Muestreo multinivel",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo requerido para hacer un muestreo multinivel"), 3),
                "peso_ahp": _safe_float(params.get("Calidad de un punto multinivel"), 0.0),
                "produce_muestra_lab": True,
            },
            {
                "actividad": "Medición 24 horas",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo que consume una medición de 24 horas"), 25),
                "peso_ahp": _safe_float(params.get("Calidad de una medición de 24 horas"), 0.0),
                "produce_muestra_lab": False,
            },
            {
                "actividad": "Medición extendida",
                "tiempo_h_por_punto": _safe_float(params.get("Tiempo que consume una medición extendida (5 días)"), 121),
                "peso_ahp": _safe_float(params.get("Calidad de una medición extendida"), 0.0),
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
            },
            {
                "analisis": "Isotopía de Deuterio",
                "costo_por_muestra": isotopic_cost,
                "peso_ahp": _safe_float(params.get("Calidad de isotopia de Deuterio"), 0.0),
            },
            {
                "analisis": "Isotopía de Helio",
                "costo_por_muestra": isotopic_cost,
                "peso_ahp": _safe_float(params.get("Calidad de isotopia de Helio"), 0.0),
            },
            {
                "analisis": "Caracterización mineralógica",
                "costo_por_muestra": _safe_float(params.get("Caracterización Mineralógica"), 0.0),
                "peso_ahp": _safe_float(params.get("Calidad de caracterizacion mineralogica"), 0.0),
            },
            {
                "analisis": "Biogeoquímica",
                "costo_por_muestra": _safe_float(params.get("Análisis Biogeoquímico"), 0.0),
                "peso_ahp": _safe_float(params.get("Calidad de biogeoquimica"), 0.0),
            },
        ]
    )

    return ParsedData(params=params, circles=circles, field_df=field_df, lab_df=lab_df)


def parse_excel(file_obj) -> ParsedData:
    """Lee un Excel en dos formatos: plantilla nueva o Data.xlsx original."""
    xls = pd.ExcelFile(file_obj)
    sheets = {s.lower().strip() for s in xls.sheet_names}
    if {"parametros", "circulos", "campo", "laboratorio"}.issubset(sheets):
        return _parse_new_template(xls)
    if {"master_datos", "sheet2"}.issubset(sheets):
        return _parse_legacy_excel(xls)
    raise ValueError(
        "Formato no reconocido. Usa la plantilla con hojas Parametros, Circulos, Campo y Laboratorio, "
        "o el Data.xlsx original con Master_Datos y Sheet2."
    )

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
# Helpers de exportación
# -----------------------------

INVALID_SHEET_CHARS = str.maketrans({c: "_" for c in '[]:*?/\\'})


def safe_sheet_name(name: str, prefix: str = "", used: Optional[set] = None) -> str:
    """Crea nombres de hoja válidos para Excel: máximo 31 caracteres y sin caracteres inválidos."""
    used = used if used is not None else set()
    base = f"{prefix}{name}".translate(INVALID_SHEET_CHARS).strip()
    base = "_".join(base.split())
    if not base:
        base = "Hoja"
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
    """Distribuye enteros proporcionalmente a capacidades, manteniendo suma exacta."""
    total = int(total)
    capacities = pd.to_numeric(capacities, errors="coerce").fillna(0).astype(float)
    if total <= 0 or capacities.sum() <= 0:
        return pd.Series([0] * len(capacities), index=capacities.index, dtype=int)

    raw = capacities / capacities.sum() * total
    base = raw.apply(math.floor).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = (raw - base).sort_values(ascending=False).index[:remainder]
        base.loc[order] += 1
    return base.astype(int)



def _integer_proportional_allocation_capped(total: int, capacities: pd.Series) -> pd.Series:
    """Distribuye enteros proporcionalmente sin superar la capacidad de cada fila."""
    total = int(total)
    capacities = pd.to_numeric(capacities, errors="coerce").fillna(0).astype(int)
    result = pd.Series([0] * len(capacities), index=capacities.index, dtype=int)
    if total <= 0 or capacities.sum() <= 0:
        return result
    if total > int(capacities.sum()):
        total = int(capacities.sum())

    raw = capacities.astype(float) / float(capacities.sum()) * total
    base = raw.apply(math.floor).astype(int)
    base = pd.concat([base, capacities], axis=1).min(axis=1).astype(int)
    result = base.copy()
    remainder = total - int(result.sum())

    fractional = (raw - base).sort_values(ascending=False)
    while remainder > 0:
        progressed = False
        for idx in fractional.index:
            if result.loc[idx] < capacities.loc[idx]:
                result.loc[idx] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            break
    return result.astype(int)


def build_lab_allocation_by_circle(plan_df: pd.DataFrame, lab_result_df: pd.DataFrame, lab_input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera una asignación sugerida de análisis de laboratorio por círculo.

    El MILP decide cantidades globales L_k. Esta función reparte esas cantidades entre
    círculos respetando la conservación total: los análisis asignados a un círculo no
    superan sus muestras físicas disponibles, salvo redondeos imposibles por datos vacíos.
    """
    if plan_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    plan = plan_df.copy()
    if "produce_muestra_lab" not in plan.columns:
        plan["produce_muestra_lab"] = False

    all_circles = plan[["circulo", "perimetro_m"]].drop_duplicates().sort_values("circulo").reset_index(drop=True)
    capacity = (
        plan[plan["produce_muestra_lab"].astype(bool)]
        .groupby("circulo", as_index=True)["puntos"]
        .sum()
        .rename("muestras_campo_disponibles")
    )
    wide = all_circles.merge(capacity, left_on="circulo", right_index=True, how="left")
    wide["muestras_campo_disponibles"] = pd.to_numeric(wide["muestras_campo_disponibles"], errors="coerce").fillna(0).astype(int)

    lab_costs = lab_input_df[["analisis", "costo_por_muestra"]].copy()
    lab_counts = lab_result_df[["analisis", "muestras_L"]].merge(lab_costs, on="analisis", how="left")
    lab_counts["muestras_L"] = pd.to_numeric(lab_counts["muestras_L"], errors="coerce").fillna(0).astype(int)

    total_lab_samples = int(lab_counts["muestras_L"].sum())
    circle_quota = _integer_proportional_allocation_capped(total_lab_samples, wide["muestras_campo_disponibles"])
    wide["lab_total_asignado"] = circle_quota.values
    remaining_by_circle = circle_quota.copy()

    long_rows = []
    # Asignar todos los análisis menos el último de forma proporcional a la cuota restante;
    # el último absorbe el saldo para mantener sumas exactas por análisis y por círculo.
    for pos, (_, lab_row) in enumerate(lab_counts.iterrows()):
        analysis = lab_row["analisis"]
        total_samples = int(lab_row["muestras_L"])
        cost_per_sample = _safe_float(lab_row.get("costo_por_muestra"), 0)
        if pos == len(lab_counts) - 1:
            allocation = remaining_by_circle.copy()
        else:
            allocation = _integer_proportional_allocation_capped(total_samples, remaining_by_circle)
        wide[analysis] = allocation.values
        remaining_by_circle = (remaining_by_circle - allocation).clip(lower=0).astype(int)
        for idx, alloc in allocation.items():
            long_rows.append(
                {
                    "circulo": wide.loc[idx, "circulo"],
                    "perimetro_m": wide.loc[idx, "perimetro_m"],
                    "muestras_campo_disponibles": int(wide.loc[idx, "muestras_campo_disponibles"]),
                    "analisis": analysis,
                    "muestras_asignadas": int(alloc),
                    "costo_estimado_COP": int(round(int(alloc) * cost_per_sample)),
                }
            )

    wide["muestras_campo_sin_usar_lab"] = wide["muestras_campo_disponibles"] - wide["lab_total_asignado"]
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = long_df.sort_values(["circulo", "analisis"]).reset_index(drop=True)
    return wide, long_df

def build_excel_report(result: Dict, field_input_df: pd.DataFrame, lab_input_df: pd.DataFrame) -> bytes:
    """Construye un Excel multi-hoja con resumen, plan por actividad y laboratorio por círculo."""
    output = io.BytesIO()
    used_sheets = set()

    kpis = result.get("kpis", {})
    summary_rows = [
        {"Indicador": key, "Valor": value}
        for key, value in kpis.items()
    ]
    summary = pd.DataFrame(summary_rows)

    plan = result.get("plan", pd.DataFrame()).copy()
    field = result.get("field", pd.DataFrame()).copy()
    lab = result.get("lab", pd.DataFrame()).copy()
    logs = pd.DataFrame(result.get("logs", []))
    lab_by_circle, lab_long = build_lab_allocation_by_circle(plan, lab, lab_input_df)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Hojas principales
        sheet = safe_sheet_name("Resumen", used=used_sheets)
        summary.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
        field.to_excel(writer, sheet_name=sheet, index=False, startrow=len(summary) + 3)
        lab.to_excel(writer, sheet_name=sheet, index=False, startrow=len(summary) + len(field) + 6)

        sheet = safe_sheet_name("Plan_completo", used=used_sheets)
        plan.to_excel(writer, sheet_name=sheet, index=False)

        sheet = safe_sheet_name("Campo_resumen", used=used_sheets)
        field.to_excel(writer, sheet_name=sheet, index=False)

        sheet = safe_sheet_name("Laboratorio_resumen", used=used_sheets)
        lab.to_excel(writer, sheet_name=sheet, index=False)

        sheet = safe_sheet_name("Lab_por_circulo", used=used_sheets)
        lab_by_circle.to_excel(writer, sheet_name=sheet, index=False)

        sheet = safe_sheet_name("Lab_detalle_circulo", used=used_sheets)
        lab_long.to_excel(writer, sheet_name=sheet, index=False)

        # Una hoja por tipo de actividad de campo
        if not plan.empty and "actividad" in plan.columns:
            for activity in sorted(plan["actividad"].dropna().unique()):
                filtered = plan[plan["actividad"] == activity].copy()
                sheet = safe_sheet_name(str(activity), prefix="Campo_", used=used_sheets)
                filtered.to_excel(writer, sheet_name=sheet, index=False)

        # Una hoja por tipo de análisis de laboratorio
        if not lab_long.empty and "analisis" in lab_long.columns:
            for analysis in sorted(lab_long["analisis"].dropna().unique()):
                filtered = lab_long[lab_long["analisis"] == analysis].copy()
                sheet = safe_sheet_name(str(analysis), prefix="Lab_", used=used_sheets)
                filtered.to_excel(writer, sheet_name=sheet, index=False)

        sheet = safe_sheet_name("Log_solver", used=used_sheets)
        logs.to_excel(writer, sheet_name=sheet, index=False)

        # Formato básico: congelar encabezados y ajustar anchuras.
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                header = col_cells[0].value if col_cells else ""
                max_len = len(str(header)) if header is not None else 0
                for cell in col_cells[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 38)

    return output.getvalue()


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
    for col in ["tiempo_h_por_punto", "peso_ahp"]:
        field_df[col] = pd.to_numeric(field_df[col], errors="coerce").fillna(0)
    for col in ["costo_por_muestra", "peso_ahp"]:
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


    # Conservación de muestras opcional:
    # el total de análisis de laboratorio no puede superar las muestras físicas producidas en campo.
    if lab_cap_enabled:
        sample_capacity = sum(Q[f] for f in F if produce_lab_sample.get(f, False))
        model.Add(sum(L[k] for k in K) <= sample_capacity)

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
                                "produce_muestra_lab": produce_lab_sample.get(f, False),
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
                "Muestras físicas para laboratorio": int(sum(q for f, q in [(row["actividad"], int(solver.Value(Q[row["actividad"]]))) for _, row in field_df.iterrows()] if produce_lab_sample.get(f, False))),
                "Muestras laboratorio totales": int(sum(int(solver.Value(L[k])) for k in K)),
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


def render_solution(result: Dict, lab_by_circle: pd.DataFrame, excel_report: bytes, csv_bytes: bytes):
    """Muestra la última solución guardada en session_state.

    Streamlit vuelve a ejecutar el script cuando se interactúa con widgets. Por eso
    guardamos la solución y los archivos de descarga en st.session_state: así, al
    descargar el Excel o CSV, la app no pierde el resultado mostrado.
    """
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
        c_csv, c_xlsx = st.columns(2)
        with c_csv:
            st.download_button(
                "Descargar plan CSV",
                data=csv_bytes,
                file_name="plan_muestreo.csv",
                mime="text/csv",
                use_container_width=True,
                on_click="ignore",
            )
        with c_xlsx:
            st.download_button(
                "Descargar reporte Excel multi-hoja",
                data=excel_report,
                file_name="reporte_muestreo_MILP_AHP.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                on_click="ignore",
            )
        st.caption("El CSV solo puede contener una tabla. Para tener varias hojas usa el reporte Excel multi-hoja.")
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
        st.markdown("**Resumen global de laboratorio**")
        st.dataframe(result["lab"], use_container_width=True)
        st.markdown("**Asignación sugerida de laboratorio por círculo**")
        st.caption(
            "El modelo optimiza las cantidades globales de laboratorio. "
            "Esta tabla reparte esas cantidades entre círculos proporcionalmente a las muestras físicas disponibles, "
            "respetando que el total de análisis no supere las muestras físicas recolectadas."
        )
        st.dataframe(lab_by_circle, use_container_width=True)
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


st.title("🧪 Optimización MILP-AHP para estrategias de muestreo")
st.caption("Asignación de recursos entre campo y laboratorio usando prioridades AHP, restricciones geométricas, presupuesto y capacidad operativa.")

with st.sidebar:
    st.header("1) Datos")
    st.download_button(
        "📥 Descargar plantilla Excel",
        data=build_template_excel_bytes(),
        file_name="Data_template_MILP_AHP.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        help="Plantilla con hojas Parametros, Circulos, Campo y Laboratorio.",
        on_click="ignore",
    )
    data_mode = st.radio(
        "Origen de datos",
        options=["Subir Excel", "Ingresar manualmente"],
        index=0,
        help="Puedes cargar un Excel con el formato de la plantilla o ingresar/editar los datos directamente en la app.",
        key="data_mode",
    )
    uploaded = None
    if data_mode == "Subir Excel":
        uploaded = st.file_uploader("Carga tu Data.xlsx", type=["xlsx"], key="uploaded_excel")

    st.header("2) Solver")
    separations_text = st.text_input("Separaciones permitidas en metros", value="10, 20, 50", key="separations_text")
    n_mode_label = st.selectbox(
        "Cálculo de puntos por perímetro",
        options=["floor", "ceil", "round"],
        index=0,
        help="floor = cuántos puntos caben sin exceder el perímetro; ceil = cubre todo el perímetro.",
        key="n_mode_label",
    )
    allow_multiple = st.checkbox(
        "Permitir varias actividades por círculo (formulación Word)",
        value=True,
        help="Más fiel al modelo: cada círculo incluido debe tener al menos una actividad, pero no necesariamente muestreo puntual.",
        key="allow_multiple",
    )
    lab_cap_enabled = st.checkbox(
        "Conservación total de muestras: laboratorio ≤ campo",
        value=True,
        help="Impone suma de todos los análisis de laboratorio ≤ muestras físicas recolectadas en actividades marcadas como Produce muestra lab.",
        key="lab_cap_enabled",
    )
    time_limit = st.slider("Tiempo máximo por etapa (s)", min_value=3, max_value=120, value=15, step=1, key="time_limit")
    relative_tolerance = st.select_slider(
        "Tolerancia lexicográfica relativa",
        options=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        value=1e-6,
        format_func=lambda x: f"{x:g}",
        key="relative_tolerance",
    )

# Carga inicial: Excel o entrada manual.
if data_mode == "Subir Excel":
    if uploaded is None:
        st.info("Carga tu archivo Data.xlsx o descarga la plantilla para llenarla.")
        if st.session_state.get("last_result"):
            st.markdown("### Última solución guardada")
            st.caption("Todavía se muestra porque está guardada en memoria de la sesión.")
            render_solution(
                st.session_state["last_result"],
                st.session_state.get("last_lab_by_circle", pd.DataFrame()),
                st.session_state.get("last_excel_report", b""),
                st.session_state.get("last_csv", b""),
            )
        st.stop()
    try:
        parsed = parse_excel(uploaded)
    except Exception as exc:
        st.error(f"No pude leer el Excel: {exc}")
        st.stop()
else:
    parsed = default_manual_data()
    st.info("Modo manual activo: edita las tablas directamente en la app. También puedes descargar la plantilla si prefieres preparar los datos en Excel.")

params = parsed.params

st.subheader("Parámetros generales")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    budget = st.number_input("Presupuesto total COP", value=int(_safe_float(params.get("Presupuesto Total Disponible"), 0)), min_value=1, step=1_000_000, key="budget")
with c2:
    days = st.number_input("Días disponibles", value=int(_safe_float(params.get("Tiempo total disponible para el muestreo"), 60)), min_value=1, step=1, key="days")
with c3:
    hours = st.number_input("Horas/día", value=int(_safe_float(params.get("Horas efectivas de trabajo del equipo por día"), 8)), min_value=1, max_value=24, step=1, key="hours")
with c4:
    max_teams = st.number_input("Equipos/sensores máx.", value=int(_safe_float(params.get("Cantidad de equipos/sensores disponibles"), 2)), min_value=1, step=1, key="max_teams")
with c5:
    logistic_daily = st.number_input("Costo logístico diario COP", value=int(_safe_float(params.get("Logística de Muestreo (Honorarios, Viáticos y Transporte)"), 0)), min_value=0, step=100_000, key="logistic_daily")
with c6:
    reserve_pct = st.number_input("Reserva/imprevistos %", value=0.0, min_value=0.0, max_value=50.0, step=1.0, help="Porcentaje del presupuesto que se deja sin optimizar para imprevistos.", key="reserve_pct")

effective_budget = int(round(float(budget) * (1 - float(reserve_pct) / 100)))
st.caption(f"Presupuesto optimizable por el solver: ${effective_budget:,.0f} COP. Reserva: ${int(budget) - effective_budget:,.0f} COP.")

st.subheader("Círculos de hadas / perímetros")
with st.expander("Ver y editar círculos", expanded=False):
    circles_df = st.data_editor(
        parsed.circles,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"Perimetro_m": st.column_config.NumberColumn("Perímetro (m)", min_value=0.01)},
        key="circles_editor",
    )

st.subheader("Actividades y análisis")
st.caption("Esta versión no usa mínimos obligatorios. Las cantidades resultan de AHP, presupuesto, geometría, capacidad operativa y conservación de muestras.")
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
            "produce_muestra_lab": st.column_config.CheckboxColumn("Produce muestra lab"),
        },
        key="field_editor",
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
        },
        key="lab_editor",
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

run = st.button("🚀 Ejecutar optimización", type="primary", use_container_width=True, key="run_optimization")

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
                budget_cop=int(effective_budget),
                days_available=int(days),
                hours_per_day=int(hours),
                max_teams=int(max_teams),
                logistic_daily_cost=int(logistic_daily),
                n_mode=n_mode_label,
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

    lab_by_circle, lab_long = build_lab_allocation_by_circle(result["plan"], result["lab"], clean_lab_df)
    excel_report = build_excel_report(result, clean_field_df, clean_lab_df)
    csv_bytes = result["plan"].to_csv(index=False).encode("utf-8")

    st.session_state["last_result"] = result
    st.session_state["last_lab_by_circle"] = lab_by_circle
    st.session_state["last_excel_report"] = excel_report
    st.session_state["last_csv"] = csv_bytes
    st.session_state["last_clean_field_df"] = clean_field_df
    st.session_state["last_clean_lab_df"] = clean_lab_df

if st.session_state.get("last_result"):
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown("## Resultado de optimización")
        st.caption("El resultado queda guardado en la sesión, por lo que descargar el Excel/CSV no borra la solución mostrada.")
    with top_right:
        if st.button("Borrar resultado guardado", use_container_width=True, key="clear_result"):
            for key in ["last_result", "last_lab_by_circle", "last_excel_report", "last_csv", "last_clean_field_df", "last_clean_lab_df"]:
                st.session_state.pop(key, None)
            st.rerun()
    render_solution(
        st.session_state["last_result"],
        st.session_state.get("last_lab_by_circle", pd.DataFrame()),
        st.session_state.get("last_excel_report", b""),
        st.session_state.get("last_csv", b""),
    )
else:
    st.markdown(
        """
        ### Flujo sugerido
        1. Sube un Excel con el formato de la plantilla o selecciona **Ingresar manualmente**.
        2. Revisa o edita parámetros generales, círculos, campo, laboratorio y pesos AHP.
        3. Define las separaciones permitidas.
        4. Ejecuta la optimización y descarga el reporte Excel multi-hoja.

        **Nota:** si el solver tarda o devuelve solo `FEASIBLE`, prueba con menos separaciones o desactiva
        “Permitir varias actividades por círculo” para usar una versión más rápida del problema.
        """
    )

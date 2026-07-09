# ruff: noqa: E501,I001

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama
import pandas as pd


def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("No se pudo localizar la raiz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd()).resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triaje_ia.config import PROMPTS_DIR  # noqa: E402
from triaje_ia.inference.predictor import TriajePredictor  # noqa: E402
from triaje_ia.llm.extractor import _completar_datos_explicitos  # noqa: E402
from triaje_ia.llm.normalizer import normalizar_vector_clinico  # noqa: E402
from triaje_ia.llm.schemas import VectorClinico  # noqa: E402
from triaje_ia.llm.validator import validar_vector_clinico  # noqa: E402


DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
OUT_BASE_DIR = PROJECT_ROOT / "reports" / "final_prompt_audit"

PROMPTS = {
    "v2_produccion": PROMPTS_DIR / "extractor_system_v2.txt",
    "v3_structured": PROMPTS_DIR / "extractor_system_v3_structured.txt",
    "v3_final": PROMPTS_DIR / "extractor_system_v3_final.txt",
    "v4_triage_cases": PROMPTS_DIR / "extractor_system_v4_triage_cases.txt",
}

SCALAR_FIELDS = [
    "edad",
    "sexo",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
    "frecuencia_respiratoria",
    "saturacion_oxigeno",
    "temperatura",
    "nivel_dolor",
]


@dataclass(frozen=True)
class FinalAuditCase:
    case_id: str
    familia_clinica: str
    esi_referencia: int
    dificultad: str
    foco_auditoria: str
    narrativa: str
    vector_esperado: dict[str, Any]
    razon_referencia: str


def _vector(
    *,
    edad: int,
    sexo: str,
    sintomas: list[str],
    patologias: list[str] | None = None,
    medicacion: list[str] | None = None,
    ta: tuple[int | None, int | None] = (None, None),
    fc: int | None = None,
    fr: int | None = None,
    sat: float | None = None,
    temp: float | None = None,
    dolor: int | None = None,
    duracion: str | None = None,
) -> dict[str, Any]:
    return {
        "edad": edad,
        "sexo": sexo,
        "sintomas_presentes": sintomas,
        "patologias_previas": patologias or [],
        "medicacion_habitual": medicacion or [],
        "presion_sistolica": ta[0],
        "presion_diastolica": ta[1],
        "frecuencia_cardiaca": fc,
        "frecuencia_respiratoria": fr,
        "saturacion_oxigeno": sat,
        "temperatura": temp,
        "nivel_dolor": dolor,
        "duracion_sintomas": duracion,
    }


def _case(
    idx: int,
    familia: str,
    esi: int,
    dificultad: str,
    foco: str,
    narrativa: str,
    vector: dict[str, Any],
) -> FinalAuditCase:
    return FinalAuditCase(
        case_id=f"F{idx:03d}",
        familia_clinica=familia,
        esi_referencia=esi,
        dificultad=dificultad,
        foco_auditoria=foco,
        narrativa=narrativa,
        vector_esperado=vector,
        razon_referencia=(
            f"Gold sintetico de auditoria final: ESI {esi} por {foco}. "
            "No procede de test real ni se usa para seleccion de modelo."
        ),
    )


def construir_casos_finales() -> list[FinalAuditCase]:
    """Cincuenta historias sinteticas con VectorClinico gold explicito."""

    raw: list[tuple[str, int, str, str, str, dict[str, Any]]] = [
        (
            "dolor_toracico",
            1,
            "critico",
            "shock cardiogenico",
            "Varon 58a traido por SAMU. Dolor toracico opresivo 10/10 desde hace 45 min, irradiado a brazo izq y mandibula. Muy palido, sudoroso, confuso por momentos. TA 78/45, FC 136, FR 32, SatO2 86%, T 36.2, Glasgow 13. DM2, HTA, fumador.",
            _vector(edad=58, sexo="M", sintomas=["chest pain", "sweating", "altered mental status"], patologias=["diabetes mellitus tipo 2", "hipertension arterial"], ta=(78, 45), fc=136, fr=32, sat=86.0, temp=36.2, dolor=10, duracion="45 min"),
        ),
        (
            "disnea",
            1,
            "critico",
            "insuficiencia respiratoria aguda",
            "Mujer 72a EPOC. Llega en silla con disnea intensa, no puede hablar frases completas. Cianosis peribucal y tiraje. TA 92/58, FC 128, FR 38, SatO2 78% aire ambiente, T 37.1, Glasgow 14.",
            _vector(edad=72, sexo="F", sintomas=["dyspnea", "cyanosis"], patologias=["EPOC"], ta=(92, 58), fc=128, fr=38, sat=78.0, temp=37.1),
        ),
        (
            "neurologia",
            1,
            "critico",
            "ictus con deterioro neurologico",
            "Varon 66a inicio brusco hace 25 min de desviacion comisura, afasia y hemiparesia derecha. Somnoliento, responde mal. TA 205/112, FC 88, FR 20, SatO2 95%, T 36.6, Glasgow 11. Anticoagulado con sintrom.",
            _vector(edad=66, sexo="M", sintomas=["aphasia", "hemiparesis", "altered mental status"], medicacion=["anticoagulants - coumarin"], ta=(205, 112), fc=88, fr=20, sat=95.0, temp=36.6, duracion="25 min"),
        ),
        (
            "sepsis",
            1,
            "critico",
            "shock septico",
            "Mujer 81a residencia, fiebre y decaimiento. Obnubilada, piel moteada, relleno capilar lento. TA 82/40, FC 122, FR 30, SatO2 89%, T 39.4, Glasgow 12. Sonda vesical, olor fuerte orina.",
            _vector(edad=81, sexo="F", sintomas=["fever", "altered mental status", "hypoxemia"], ta=(82, 40), fc=122, fr=30, sat=89.0, temp=39.4),
        ),
        (
            "alergia",
            1,
            "critico",
            "anafilaxia",
            "Mujer 29a tras comer frutos secos. Urticaria generalizada, edema labios y lengua, estridor y mareo. TA 75/42, FC 132, FR 30, SatO2 88%, T 36.7, Glasgow 14. Alergia conocida a nueces.",
            _vector(edad=29, sexo="F", sintomas=["urticaria", "angioedema", "stridor", "dizziness"], patologias=["alergia a nueces"], ta=(75, 42), fc=132, fr=30, sat=88.0, temp=36.7),
        ),
        (
            "endocrino",
            1,
            "critico",
            "hipoglucemia grave",
            "Varon 76a diabetico insulinodependiente, encontrado sudoroso y confuso. Glucemia capilar 32 mg/dl. TA 118/70, FC 104, FR 22, SatO2 96%, T 36.1, Glasgow 10. No ha comido hoy.",
            _vector(edad=76, sexo="M", sintomas=["altered mental status", "hypoglycemia", "sweating"], patologias=["diabetes mellitus"], medicacion=["insulin analogs"], ta=(118, 70), fc=104, fr=22, sat=96.0, temp=36.1),
        ),
        (
            "convulsion",
            1,
            "critico",
            "convulsion activa",
            "Varon 45a entra convulsionando, crisis generalizada activa de varios minutos segun familia. Mordedura lengua, cianosis leve. TA no registrable inicial, FC 150, FR irregular, SatO2 82%, T 37.0, Glasgow no valorable durante crisis. Epilepsia conocida.",
            _vector(edad=45, sexo="M", sintomas=["seizure", "cyanosis"], patologias=["epilepsia"], ta=(None, None), fc=150, fr=None, sat=82.0, temp=37.0, duracion="varios minutos"),
        ),
        (
            "trauma",
            1,
            "critico",
            "politrauma inestable",
            "Varon 34a accidente moto alta energia. Dolor toracico y abdominal, muy agitado. TA 84/50, FC 145, FR 34, SatO2 90%, T 35.8, Glasgow 12. Deformidad femur izq y herida sangrante en muslo.",
            _vector(edad=34, sexo="M", sintomas=["chest pain", "abdominal pain", "altered mental status", "bleeding"], ta=(84, 50), fc=145, fr=34, sat=90.0, temp=35.8),
        ),
        (
            "intoxicacion",
            1,
            "critico",
            "coma toxico",
            "Mujer 38a encontrada con blisters vacios de benzodiacepinas. Somnolencia profunda, responde solo al dolor. TA 90/55, FC 54, FR 8, SatO2 84%, T 35.9, Glasgow 7. Olor alcohol.",
            _vector(edad=38, sexo="F", sintomas=["altered mental status", "bradypnea"], medicacion=["benzodiazepines"], ta=(90, 55), fc=54, fr=8, sat=84.0, temp=35.9),
        ),
        (
            "hemorragia",
            1,
            "critico",
            "hemorragia digestiva masiva",
            "Varon 69a cirrosis. Hematemesis abundante, palido, sudoroso, mareado. TA 70/38, FC 140, FR 28, SatO2 91%, T 36.0, Glasgow 13. Abdomen distendido.",
            _vector(edad=69, sexo="M", sintomas=["hematemesis", "dizziness", "sweating"], patologias=["cirrosis"], ta=(70, 38), fc=140, fr=28, sat=91.0, temp=36.0),
        ),
        (
            "dolor_toracico",
            2,
            "alto",
            "dolor toracico tiempo-dependiente",
            "Varon 67a acompanado por su mujer, dolor pecho 1h, opresivo, sube a brazo izq. Sudoroso, nauseas, mareado. TA 158/94, FC 102, FR 22, SatO2 94%, T 36.8, Glasgow 15. EVA 8/10. HTA y DM.",
            _vector(edad=67, sexo="M", sintomas=["chest pain", "sweating", "nausea", "dizziness"], patologias=["hipertension arterial", "diabetes mellitus"], ta=(158, 94), fc=102, fr=22, sat=94.0, temp=36.8, dolor=8, duracion="1h"),
        ),
        (
            "neurologia",
            2,
            "alto",
            "ictus tiempo-dependiente",
            "Mujer 74a habla rara desde hace 50 min, fuerza disminuida brazo izq. Orientada pero ansiosa. TA 188/96, FC 82, FR 18, SatO2 96%, T 36.5, Glasgow 15. Toma apixaban.",
            _vector(edad=74, sexo="F", sintomas=["aphasia", "arm weakness", "anxiety"], medicacion=["direct factor Xa inhibitors"], ta=(188, 96), fc=82, fr=18, sat=96.0, temp=36.5, duracion="50 min"),
        ),
        (
            "disnea",
            2,
            "alto",
            "asma moderada grave",
            "Mujer 22a asmatica, disnea y pitos desde la noche. Habla entrecortado, uso de salbutamol repetido. TA 126/78, FC 118, FR 30, SatO2 91%, T 36.9, Glasgow 15. No fiebre.",
            _vector(edad=22, sexo="F", sintomas=["dyspnea", "wheezing"], patologias=["asma"], medicacion=["bronchodilators - short acting"], ta=(126, 78), fc=118, fr=30, sat=91.0, temp=36.9, duracion="desde la noche"),
        ),
        (
            "dolor_abdominal",
            2,
            "alto",
            "abdomen agudo",
            "Varon 61a dolor abdominal intenso inicio brusco, abdomen duro, palido. TA 100/62, FC 116, FR 24, SatO2 96%, T 37.8, Glasgow 15. EVA 9/10. No vomitos.",
            _vector(edad=61, sexo="M", sintomas=["abdominal pain", "pallor"], ta=(100, 62), fc=116, fr=24, sat=96.0, temp=37.8, dolor=9),
        ),
        (
            "sepsis",
            2,
            "alto",
            "sepsis sin shock claro",
            "Mujer 58a fiebre 39, escalofrios, tos productiva y mal estado general. TA 104/66, FC 124, FR 28, SatO2 92%, T 39.2, Glasgow 15. Diabetica.",
            _vector(edad=58, sexo="F", sintomas=["fever", "chills", "cough"], patologias=["diabetes mellitus"], ta=(104, 66), fc=124, fr=28, sat=92.0, temp=39.2),
        ),
        (
            "trauma",
            2,
            "alto",
            "trauma craneal anticoagulado",
            "Varon 83a caida en domicilio, golpe cabeza, herida cuero cabelludo. Toma sintrom. No perdida conciencia clara, algo desorientado. TA 146/82, FC 92, FR 18, SatO2 96%, T 36.4, Glasgow 14. Dolor cabeza 6/10.",
            _vector(edad=83, sexo="M", sintomas=["head injury", "scalp wound", "altered mental status", "headache"], medicacion=["anticoagulants - coumarin"], ta=(146, 82), fc=92, fr=18, sat=96.0, temp=36.4, dolor=6),
        ),
        (
            "psiquiatria",
            2,
            "alto",
            "riesgo autolitico alto",
            "Mujer 31a acude con pareja por ideas autoliticas activas y plan con medicacion. Llanto, muy angustiada. TA 122/74, FC 104, FR 20, SatO2 98%, T 36.6, Glasgow 15. Antecedente depresion.",
            _vector(edad=31, sexo="F", sintomas=["suicidal ideation", "anxiety"], patologias=["depresion"], ta=(122, 74), fc=104, fr=20, sat=98.0, temp=36.6),
        ),
        (
            "embarazo",
            2,
            "alto",
            "hemorragia obstetrica",
            "Mujer 32a gestante 28 semanas, sangrado vaginal abundante y dolor abdominal tipo contraccion. Mareada. TA 98/60, FC 118, FR 22, SatO2 97%, T 36.7, Glasgow 15. Dolor 8/10.",
            _vector(edad=32, sexo="F", sintomas=["vaginal bleeding", "abdominal pain", "dizziness"], patologias=["embarazo"], ta=(98, 60), fc=118, fr=22, sat=97.0, temp=36.7, dolor=8),
        ),
        (
            "alergia",
            2,
            "alto",
            "reaccion alergica con edema",
            "Varon 40a tras antibiotico, urticaria extensa y sensacion de garganta rara. No estridor. TA 118/72, FC 105, FR 22, SatO2 96%, T 36.8, Glasgow 15. Labios algo hinchados. Sin hipotension.",
            _vector(edad=40, sexo="M", sintomas=["urticaria", "angioedema"], ta=(118, 72), fc=105, fr=22, sat=96.0, temp=36.8),
        ),
        (
            "cefalea",
            2,
            "alto",
            "cefalea trueno",
            "Mujer 47a cefalea brusca maxima intensidad, peor dolor de su vida. Vomitos y fotofobia. TA 176/98, FC 88, FR 18, SatO2 97%, T 36.4, Glasgow 15. Rigidez nuca leve. EVA 10/10.",
            _vector(edad=47, sexo="F", sintomas=["thunderclap headache", "vomiting", "photophobia", "neck stiffness"], ta=(176, 98), fc=88, fr=18, sat=97.0, temp=36.4, dolor=10),
        ),
        (
            "dolor_toracico",
            3,
            "medio",
            "dolor toracico estable",
            "Varon 45a dolor toracico punzante al respirar desde ayer. TA 132/78, FC 88, FR 18, SatO2 98%, T 36.6, Glasgow 15. Dolor 5/10. Sin sudoracion, sin disnea.",
            _vector(edad=45, sexo="M", sintomas=["chest pain"], ta=(132, 78), fc=88, fr=18, sat=98.0, temp=36.6, dolor=5, duracion="desde ayer"),
        ),
        (
            "digestivo",
            3,
            "medio",
            "dolor FID",
            "Mujer 54a dolor FID de 12 horas, nauseas y un vomito. TA 118/76, FC 92, FR 16, SatO2 98%, T 37.8, Glasgow 15. Dolor 7/10. DM2 con metformina.",
            _vector(edad=54, sexo="F", sintomas=["right lower quadrant abdominal pain", "nausea", "vomiting"], patologias=["diabetes mellitus tipo 2"], medicacion=["biguanides"], ta=(118, 76), fc=92, fr=16, sat=98.0, temp=37.8, dolor=7, duracion="12 horas"),
        ),
        (
            "urinario",
            3,
            "medio",
            "pielonefritis posible",
            "Mujer 63a fiebre, escalofrios y dolor lumbar derecho desde ayer. Disuria. TA 126/74, FC 108, FR 20, SatO2 97%, T 38.6, Glasgow 15. Dolor 6/10.",
            _vector(edad=63, sexo="F", sintomas=["fever", "chills", "flank pain", "dysuria"], ta=(126, 74), fc=108, fr=20, sat=97.0, temp=38.6, dolor=6, duracion="desde ayer"),
        ),
        (
            "syncope",
            3,
            "medio",
            "sincope recuperado",
            "Varon 70a sincope breve en supermercado, ahora consciente y orientado. Mareo previo. TA 110/68, FC 58, FR 16, SatO2 97%, T 36.4, Glasgow 15. Toma bisoprolol.",
            _vector(edad=70, sexo="M", sintomas=["syncope", "dizziness"], medicacion=["beta blockers cardiac selective"], ta=(110, 68), fc=58, fr=16, sat=97.0, temp=36.4),
        ),
        (
            "trauma",
            3,
            "medio",
            "fractura probable",
            "Mujer 68a caida con deformidad muneca derecha y dolor intenso. No golpe cabeza. TA 142/80, FC 96, FR 18, SatO2 98%, T 36.5, Glasgow 15. Dolor 8/10.",
            _vector(edad=68, sexo="F", sintomas=["wrist pain", "deformity"], ta=(142, 80), fc=96, fr=18, sat=98.0, temp=36.5, dolor=8),
        ),
        (
            "respiratorio",
            3,
            "medio",
            "neumonia estable",
            "Varon 70a fiebre, tos y disnea progresiva. TA 118/70, FC 110, FR 28, SatO2 89%, T 38.7, Glasgow 15. EPOC, usa inhaladores.",
            _vector(edad=70, sexo="M", sintomas=["fever", "cough", "dyspnea"], patologias=["EPOC"], ta=(118, 70), fc=110, fr=28, sat=89.0, temp=38.7),
        ),
        (
            "ginecologia",
            3,
            "medio",
            "dolor pelvico",
            "Mujer 26a dolor hipogastrico y sangrado vaginal escaso desde esta manana. TA 116/72, FC 94, FR 18, SatO2 99%, T 37.1, Glasgow 15. Dolor 6/10.",
            _vector(edad=26, sexo="F", sintomas=["pelvic pain", "vaginal bleeding"], ta=(116, 72), fc=94, fr=18, sat=99.0, temp=37.1, dolor=6, duracion="esta manana"),
        ),
        (
            "neurologia",
            3,
            "medio",
            "vertigo intenso",
            "Varon 59a vertigo intenso con vomitos desde la noche, no deficit focal. TA 150/86, FC 84, FR 18, SatO2 98%, T 36.5, Glasgow 15. Dolor 0/10.",
            _vector(edad=59, sexo="M", sintomas=["vertigo", "vomiting"], ta=(150, 86), fc=84, fr=18, sat=98.0, temp=36.5, dolor=0, duracion="desde la noche"),
        ),
        (
            "digestivo",
            3,
            "medio",
            "rectorragia con anticoagulacion",
            "Mujer 79a rectorragia desde ayer, varias deposiciones con sangre roja. Toma acenocumarol. TA 116/70, FC 104, FR 18, SatO2 97%, T 36.6, Glasgow 15. Sin dolor.",
            _vector(edad=79, sexo="F", sintomas=["rectal bleeding"], medicacion=["anticoagulants - coumarin"], ta=(116, 70), fc=104, fr=18, sat=97.0, temp=36.6, dolor=None, duracion="desde ayer"),
        ),
        (
            "pediatria",
            3,
            "medio",
            "fiebre pediatrica",
            "Nino 5a fiebre 39 y tos, buen estado general, juega a ratos. TA 98/60, FC 132, FR 26, SatO2 97%, T 39.0, Glasgow 15 adaptado. Come menos pero bebe.",
            _vector(edad=5, sexo="M", sintomas=["fever", "cough"], ta=(98, 60), fc=132, fr=26, sat=97.0, temp=39.0),
        ),
        (
            "urologia",
            3,
            "medio",
            "retencion urinaria",
            "Varon 72a no puede orinar desde hace 1 dia, dolor suprapubico. HTA. TA 146/84, FC 96, FR 18, SatO2 98%, T 36.8, Glasgow 15. Dolor 7/10.",
            _vector(edad=72, sexo="M", sintomas=["urinary retention", "suprapubic pain"], patologias=["hipertension arterial"], ta=(146, 84), fc=96, fr=18, sat=98.0, temp=36.8, dolor=7, duracion="1 dia"),
        ),
        (
            "piel",
            3,
            "medio",
            "celulitis con fiebre",
            "Mujer 64a pierna izquierda roja, caliente y dolorosa desde ayer, fiebre. TA 132/76, FC 108, FR 20, SatO2 97%, T 38.3, Glasgow 15. Dolor 6/10.",
            _vector(edad=64, sexo="F", sintomas=["leg redness", "leg pain", "fever"], ta=(132, 76), fc=108, fr=20, sat=97.0, temp=38.3, dolor=6, duracion="desde ayer"),
        ),
        (
            "oftalmologia",
            3,
            "medio",
            "perdida visual",
            "Varon 62a perdida vision ojo derecho brusca hace 2 horas, sin dolor. TA 168/90, FC 78, FR 16, SatO2 98%, T 36.6, Glasgow 15. DM2.",
            _vector(edad=62, sexo="M", sintomas=["vision loss"], patologias=["diabetes mellitus tipo 2"], ta=(168, 90), fc=78, fr=16, sat=98.0, temp=36.6, dolor=None, duracion="2 horas"),
        ),
        (
            "trauma",
            4,
            "bajo",
            "esguince leve",
            "Mujer 19a torcedura tobillo leve, camina con dolor. TA 112/68, FC 82, FR 16, SatO2 99%, T 36.5, Glasgow 15. Dolor 4/10, sin deformidad.",
            _vector(edad=19, sexo="F", sintomas=["ankle sprain", "ankle pain"], ta=(112, 68), fc=82, fr=16, sat=99.0, temp=36.5, dolor=4),
        ),
        (
            "herida",
            4,
            "bajo",
            "herida simple",
            "Varon 27a corte superficial dedo con cuchillo cocina. Sangrado minimo controlado. TA 118/70, FC 76, FR 16, SatO2 99%, T 36.6, Glasgow 15. Dolor 3/10. Vacuna tetanos dudosa.",
            _vector(edad=27, sexo="M", sintomas=["minor wound", "bleeding"], ta=(118, 70), fc=76, fr=16, sat=99.0, temp=36.6, dolor=3),
        ),
        (
            "otorrino",
            4,
            "bajo",
            "otalgia",
            "Mujer 34a dolor de oido derecho desde ayer, febricula. Sin dolor dental. TA 122/70, FC 84, FR 16, SatO2 99%, T 37.5.",
            _vector(edad=34, sexo="F", sintomas=["ear pain", "fever"], ta=(122, 70), fc=84, fr=16, sat=99.0, temp=37.5, duracion="desde ayer"),
        ),
        (
            "dental",
            4,
            "bajo",
            "dolor dental",
            "Mujer 51a dolor muela desde hace 3 dias, no fiebre, cara sin hinchazon importante. TA 130/78, FC 84, FR 16, SatO2 99%, T 36.7, Glasgow 15. Dolor 6/10.",
            _vector(edad=51, sexo="F", sintomas=["dental pain"], ta=(130, 78), fc=84, fr=16, sat=99.0, temp=36.7, dolor=6, duracion="3 dias"),
        ),
        (
            "oftalmologia",
            4,
            "bajo",
            "conjuntivitis",
            "Mujer 34a ojo rojo y leganas desde ayer, picor. TA 116/68, FC 78, FR 16, SatO2 99%, T 36.7, Glasgow 15. No dolor intenso, no perdida vision.",
            _vector(edad=34, sexo="F", sintomas=["conjunctivitis", "itching"], ta=(116, 68), fc=78, fr=16, sat=99.0, temp=36.7, duracion="desde ayer"),
        ),
        (
            "piel",
            4,
            "bajo",
            "una encarnada",
            "Varon 24a dolor dedo gordo pie por una encarnada, rojo local. TA 118/70, FC 80, FR 16, SatO2 99%, T 36.6, Glasgow 15. No fiebre.",
            _vector(edad=24, sexo="M", sintomas=["ingrown toenail", "toe pain", "redness"], ta=(118, 70), fc=80, fr=16, sat=99.0, temp=36.6),
        ),
        (
            "piel",
            4,
            "bajo",
            "quemadura pequena",
            "Mujer 33a quemadura pequena mano con aceite, ampolla 2 cm. TA 116/70, FC 82, FR 16, SatO2 99%, T 36.5, Glasgow 15. Dolor 5/10, no circunferencial.",
            _vector(edad=33, sexo="F", sintomas=["minor burn"], ta=(116, 70), fc=82, fr=16, sat=99.0, temp=36.5, dolor=5),
        ),
        (
            "gastro",
            4,
            "bajo",
            "nauseas leves",
            "Mujer 42a nauseas sin vomitos tras comida copiosa. TA 120/72, FC 78, FR 16, SatO2 99%, T 36.6, Glasgow 15. Dolor abdominal 2/10, buen estado.",
            _vector(edad=42, sexo="F", sintomas=["nausea", "abdominal pain"], ta=(120, 72), fc=78, fr=16, sat=99.0, temp=36.6, dolor=2),
        ),
        (
            "musculoesqueletico",
            4,
            "bajo",
            "dolor hombro no traumatico",
            "Mujer 49a dolor hombro derecho desde hace una semana, empeora al levantar brazo. TA 128/76, FC 78, FR 16, SatO2 99%, T 36.5, Glasgow 15. Dolor 4/10. Sin trauma.",
            _vector(edad=49, sexo="F", sintomas=["right shoulder pain"], ta=(128, 76), fc=78, fr=16, sat=99.0, temp=36.5, dolor=4, duracion="una semana"),
        ),
        (
            "administrativo",
            5,
            "muy_bajo",
            "receta",
            "Varon 55a solicita receta de medicacion habitual porque se le acabo. Niega sintomas. TA 130/76, FC 74, FR 16, SatO2 98%, T 36.5, Glasgow 15.",
            _vector(edad=55, sexo="M", sintomas=["medication refill"], ta=(130, 76), fc=74, fr=16, sat=98.0, temp=36.5),
        ),
        (
            "administrativo",
            5,
            "muy_bajo",
            "informe de baja",
            "Varon 30a acude para pedir informe de baja por dolor lumbar ya valorado en centro salud. Sin dolor actual importante. TA 118/70, FC 72, FR 16, SatO2 99%, T 36.5, Glasgow 15.",
            _vector(edad=30, sexo="M", sintomas=["administrative request"], ta=(118, 70), fc=72, fr=16, sat=99.0, temp=36.5),
        ),
        (
            "cura",
            5,
            "muy_bajo",
            "revision de herida",
            "Mujer 46a revision herida quirurgica limpia, sin fiebre ni dolor. TA 124/74, FC 76, FR 16, SatO2 99%, T 36.4, Glasgow 15.",
            _vector(edad=46, sexo="F", sintomas=["wound check"], ta=(124, 74), fc=76, fr=16, sat=99.0, temp=36.4),
        ),
        (
            "cura",
            5,
            "muy_bajo",
            "retirada de puntos",
            "Varon 39a viene para retirada de puntos en antebrazo. Herida seca, sin signos infeccion. TA 122/76, FC 70, FR 16, SatO2 99%, T 36.5.",
            _vector(edad=39, sexo="M", sintomas=["suture removal"], ta=(122, 76), fc=70, fr=16, sat=99.0, temp=36.5),
        ),
        (
            "oftalmologia",
            5,
            "muy_bajo",
            "ojo seco",
            "Mujer 68a sequedad ocular cronica, pide gotas porque se le acabaron. Sin dolor ni perdida vision. TA 132/76, FC 72, FR 16, SatO2 98%, T 36.4.",
            _vector(edad=68, sexo="F", sintomas=["dry eye"], ta=(132, 76), fc=72, fr=16, sat=98.0, temp=36.4),
        ),
        (
            "otorrino",
            5,
            "muy_bajo",
            "oido taponado",
            "Mujer 67a sensacion oido taponado y disminucion audicion progresiva. TA 136/78, FC 76, FR 16, SatO2 98%, T 36.5, Glasgow 15. Sin dolor intenso.",
            _vector(edad=67, sexo="F", sintomas=["ear fullness", "hearing loss"], ta=(136, 78), fc=76, fr=16, sat=98.0, temp=36.5),
        ),
        (
            "musculoesqueletico",
            5,
            "muy_bajo",
            "dolor cronico rodilla",
            "Varon 73a dolor rodilla derecha cronico, sin traumatismo reciente, viene porque quiere adelantar traumatologia. TA 138/80, FC 74, FR 16, SatO2 98%, T 36.5. Dolor 3/10.",
            _vector(edad=73, sexo="M", sintomas=["chronic knee pain"], ta=(138, 80), fc=74, fr=16, sat=98.0, temp=36.5, dolor=3),
        ),
        (
            "administrativo",
            5,
            "muy_bajo",
            "certificado",
            "Mujer 28a solicita justificante laboral tras episodio de gastroenteritis ya resuelto. Niega sintomas actuales. TA 112/70, FC 72, FR 16, SatO2 99%, T 36.6.",
            _vector(edad=28, sexo="F", sintomas=["administrative request"], ta=(112, 70), fc=72, fr=16, sat=99.0, temp=36.6),
        ),
    ]

    return [_case(idx, *item) for idx, item in enumerate(raw, start=1)]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[f"{item['prompt_id']}::{item['case_id']}"] = item
    return cache


def normalize_term(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def term_set(values: list[str]) -> set[str]:
    return {normalize_term(value) for value in values if normalize_term(value)}


def scalar_match(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 0.05
        except (TypeError, ValueError):
            return False
    return expected == actual


def compare_vector(expected: dict[str, Any], actual: VectorClinico) -> dict[str, Any]:
    actual_data = actual.model_dump()
    field_matches = {
        f"{field}_match": scalar_match(expected.get(field), actual_data.get(field))
        for field in SCALAR_FIELDS
    }

    expected_symptoms = term_set(expected["sintomas_presentes"])
    actual_symptoms = term_set(actual_data["sintomas_presentes"])
    expected_pathologies = term_set(expected["patologias_previas"])
    actual_pathologies = term_set(actual_data["patologias_previas"])
    expected_meds = term_set(expected["medicacion_habitual"])
    actual_meds = term_set(actual_data["medicacion_habitual"])

    symptom_hits = expected_symptoms & actual_symptoms
    symptom_extra = actual_symptoms - expected_symptoms
    pathology_hits = expected_pathologies & actual_pathologies
    med_hits = expected_meds & actual_meds

    scalar_ok = sum(field_matches.values())
    return {
        **field_matches,
        "scalar_matches": int(scalar_ok),
        "scalar_total": len(SCALAR_FIELDS),
        "scalar_match_rate": scalar_ok / len(SCALAR_FIELDS),
        "expected_symptoms": "|".join(sorted(expected_symptoms)),
        "actual_symptoms": "|".join(sorted(actual_symptoms)),
        "symptom_hits": "|".join(sorted(symptom_hits)),
        "symptom_missing": "|".join(sorted(expected_symptoms - actual_symptoms)),
        "symptom_extra": "|".join(sorted(symptom_extra)),
        "symptom_recall": (
            len(symptom_hits) / len(expected_symptoms) if expected_symptoms else 1.0
        ),
        "symptom_extra_count": len(symptom_extra),
        "pathology_recall": (
            len(pathology_hits) / len(expected_pathologies)
            if expected_pathologies
            else 1.0
        ),
        "medication_recall": len(med_hits) / len(expected_meds) if expected_meds else 1.0,
        "empty_symptoms": len(actual_symptoms) == 0,
    }


def extract_with_prompt(
    narrativa: str,
    *,
    prompt_text: str,
    model: str,
) -> tuple[VectorClinico, VectorClinico, float]:
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Narrativa:\n{narrativa}"},
        ],
        format=VectorClinico.model_json_schema(),
        options={"temperature": 0.0},
    )
    elapsed = time.perf_counter() - start
    raw_vector = normalizar_vector_clinico(
        VectorClinico.model_validate_json(response.message.content)
    )
    final_vector = normalizar_vector_clinico(
        _completar_datos_explicitos(narrativa, raw_vector)
    )
    return raw_vector, final_vector, elapsed


def process_case(
    case: FinalAuditCase,
    *,
    prompt_id: str,
    prompt_path: Path,
    prompt_text: str,
    model: str,
    predictor: TriajePredictor | None,
) -> dict[str, Any]:
    base = {
        "prompt_id": prompt_id,
        "prompt_path": str(prompt_path.relative_to(PROJECT_ROOT)),
        "model": model,
        **asdict(case),
        "vector_esperado_json": json.dumps(case.vector_esperado, ensure_ascii=False),
    }
    try:
        raw_vector, final_vector, elapsed = extract_with_prompt(
            case.narrativa,
            prompt_text=prompt_text,
            model=model,
        )
        alertas = validar_vector_clinico(final_vector, case.narrativa)
        comparison = compare_vector(case.vector_esperado, final_vector)
        row = {
            **base,
            "ok": True,
            "error": "",
            "elapsed_seconds": round(elapsed, 3),
            "n_alertas_validador": len(alertas),
            "alertas_validador": " | ".join(str(alerta) for alerta in alertas),
            "raw_vector_json": json.dumps(raw_vector.model_dump(), ensure_ascii=False),
            "final_vector_json": json.dumps(final_vector.model_dump(), ensure_ascii=False),
            **comparison,
        }
        if predictor is not None:
            result = predictor.predict(final_vector, case.narrativa)
            probas = [float(x) for x in result.probas]
            delta = result.clase_predicha - case.esi_referencia
            row.update(
                {
                    "clase_predicha": result.clase_predicha,
                    "confianza": result.confianza,
                    "alerta_a1_activada": result.alerta_a1_activada,
                    "p_esi_1": probas[0],
                    "p_esi_2": probas[1],
                    "p_esi_3": probas[2],
                    "p_esi_4": probas[3],
                    "p_esi_5": probas[4],
                    "delta_vs_referencia": delta,
                    "abs_delta_vs_referencia": abs(delta),
                    "discrepancia_severa": abs(delta) >= 2,
                    "subtriaje": delta > 0,
                    "sobretriaje": delta < 0,
                    "fallo_seguridad_a1": case.esi_referencia == 1
                    and result.clase_predicha != 1
                    and not result.alerta_a1_activada,
                    "sobretriaje_leve_a1": case.esi_referencia >= 4
                    and result.clase_predicha == 1,
                }
            )
        return row
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prompt_id, group in results.groupby("prompt_id", sort=False):
        ok = group[group["ok"] == True].copy()  # noqa: E712
        row: dict[str, Any] = {
            "prompt_id": prompt_id,
            "n_cases": int(len(group)),
            "n_ok": int(len(ok)),
            "n_failed": int((group["ok"] != True).sum()),  # noqa: E712
        }
        if not ok.empty:
            row.update(
                {
                    "mean_elapsed_seconds": round(float(ok["elapsed_seconds"].mean()), 3),
                    "mean_scalar_match_rate": round(float(ok["scalar_match_rate"].mean()), 3),
                    "mean_symptom_recall": round(float(ok["symptom_recall"].mean()), 3),
                    "empty_symptoms": int(ok["empty_symptoms"].sum()),
                    "validator_alerts_total": int(ok["n_alertas_validador"].sum()),
                    "symptom_extra_total": int(ok["symptom_extra_count"].sum()),
                }
            )
            if "clase_predicha" in ok:
                row.update(
                    {
                        "mean_abs_delta_vs_reference": round(
                            float(ok["abs_delta_vs_referencia"].mean()), 3
                        ),
                        "severe_discrepancies": int(ok["discrepancia_severa"].sum()),
                        "safety_failures_a1": int(ok["fallo_seguridad_a1"].sum()),
                        "low_acuity_predicted_a1": int(ok["sobretriaje_leve_a1"].sum()),
                        "a1_alerts": int(ok["alerta_a1_activada"].sum()),
                        "pred_esi_1": int((ok["clase_predicha"] == 1).sum()),
                        "pred_esi_2": int((ok["clase_predicha"] == 2).sum()),
                        "pred_esi_3": int((ok["clase_predicha"] == 3).sum()),
                        "pred_esi_4": int((ok["clase_predicha"] == 4).sum()),
                        "pred_esi_5": int((ok["clase_predicha"] == 5).sum()),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    rows = [list(text.columns), *text.values.tolist()]
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(str(rows[0][i]).ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(widths))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, sep, *body])


def write_report(path: Path, summary: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# Auditoria final de prompts y flujo de produccion",
        "",
        "Auditoria sintetica con 50 historias realistas y gold explicito de VectorClinico.",
        "No modifica prompts, modelo, umbrales ni politica A1.",
        "",
        "## Resumen por prompt",
        "",
        markdown_table(summary),
        "",
        "## Errores de ejecucion",
        "",
    ]
    failed = results[results["ok"] != True]  # noqa: E712
    if failed.empty:
        lines.append("No se registraron errores de ejecucion.")
    else:
        lines.append(markdown_table(failed[["prompt_id", "case_id", "error"]]))

    ok = results[results["ok"] == True].copy()  # noqa: E712
    if not ok.empty:
        extraction_cols = [
            "prompt_id",
            "case_id",
            "familia_clinica",
            "foco_auditoria",
            "scalar_match_rate",
            "symptom_recall",
            "symptom_missing",
            "symptom_extra",
            "n_alertas_validador",
        ]
        worst_extraction = ok.sort_values(
            ["symptom_recall", "scalar_match_rate", "n_alertas_validador"],
            ascending=[True, True, False],
        ).head(20)
        lines.extend(["", "## Peores extracciones", ""])
        lines.append(markdown_table(worst_extraction[extraction_cols]))

        if "discrepancia_severa" in ok:
            severe = ok[ok["discrepancia_severa"] == True].copy()  # noqa: E712
            lines.extend(["", "## Discrepancias severas del modelo", ""])
            if severe.empty:
                lines.append("No se registraron discrepancias severas.")
            else:
                cols = [
                    "prompt_id",
                    "case_id",
                    "familia_clinica",
                    "esi_referencia",
                    "clase_predicha",
                    "confianza",
                    "alerta_a1_activada",
                    "foco_auditoria",
                ]
                lines.append(markdown_table(severe[cols]))

            safety = ok[(ok["fallo_seguridad_a1"] == True) | (ok["sobretriaje_leve_a1"] == True)].copy()  # noqa: E712
            lines.extend(["", "## Casos de seguridad a revisar", ""])
            if safety.empty:
                lines.append("No se registraron fallos A1 sin alerta ni sobretriaje leve a A1.")
            else:
                cols = [
                    "prompt_id",
                    "case_id",
                    "esi_referencia",
                    "clase_predicha",
                    "p_esi_1",
                    "alerta_a1_activada",
                    "foco_auditoria",
                ]
                lines.append(markdown_table(safety[cols]))

    path.write_text("\n".join(lines), encoding="utf-8")


def write_errors_report(path: Path, results: pd.DataFrame) -> None:
    lines = [
        "# Patrones de error de la auditoria final",
        "",
        "Este informe separa errores de extraccion, validacion y modelo. "
        "Las discrepancias se calculan contra gold sintetico, no contra test real.",
        "",
    ]

    failed = results[results["ok"] != True]  # noqa: E712
    lines.extend(["## Fallos de ejecucion", ""])
    if failed.empty:
        lines.append("No hubo fallos de ejecucion.")
    else:
        lines.append(markdown_table(failed[["prompt_id", "case_id", "error"]]))

    ok = results[results["ok"] == True].copy()  # noqa: E712
    if ok.empty:
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    extraction_flags = ok[
        (ok["symptom_recall"] < 0.5)
        | (ok["scalar_match_rate"] < 1.0)
        | (ok["n_alertas_validador"] > 0)
        | (ok["symptom_extra_count"] >= 2)
    ].copy()
    lines.extend(["", "## Extraccion LLM a revisar", ""])
    if extraction_flags.empty:
        lines.append("No se detectaron extracciones llamativas por las reglas automaticas.")
    else:
        cols = [
            "prompt_id",
            "case_id",
            "familia_clinica",
            "foco_auditoria",
            "scalar_match_rate",
            "symptom_recall",
            "symptom_missing",
            "symptom_extra",
            "alertas_validador",
        ]
        lines.append(markdown_table(extraction_flags[cols].head(60)))

    if "clase_predicha" in ok:
        severe = ok[ok["discrepancia_severa"] == True].copy()  # noqa: E712
        lines.extend(["", "## Discrepancias severas del modelo", ""])
        if severe.empty:
            lines.append("No hubo discrepancias severas.")
        else:
            cols = [
                "prompt_id",
                "case_id",
                "familia_clinica",
                "esi_referencia",
                "clase_predicha",
                "delta_vs_referencia",
                "p_esi_1",
                "confianza",
                "foco_auditoria",
            ]
            lines.append(markdown_table(severe[cols]))

        safety = ok[
            (ok["fallo_seguridad_a1"] == True)  # noqa: E712
            | (ok["sobretriaje_leve_a1"] == True)  # noqa: E712
        ].copy()
        lines.extend(["", "## Seguridad A1", ""])
        if safety.empty:
            lines.append("No hubo fallos A1 sin alerta ni casos leves enviados a A1.")
        else:
            cols = [
                "prompt_id",
                "case_id",
                "esi_referencia",
                "clase_predicha",
                "p_esi_1",
                "alerta_a1_activada",
                "foco_auditoria",
            ]
            lines.append(markdown_table(safety[cols]))

        by_case = (
            ok.groupby(["case_id", "familia_clinica", "foco_auditoria"], as_index=False)
            .agg(
                prompts_afectados=("discrepancia_severa", "sum"),
                peor_delta=("abs_delta_vs_referencia", "max"),
                media_delta_abs=("abs_delta_vs_referencia", "mean"),
            )
            .sort_values(
                ["prompts_afectados", "peor_delta", "media_delta_abs"],
                ascending=[False, False, False],
            )
        )
        lines.extend(["", "## Casos con patron transversal", ""])
        lines.append(markdown_table(by_case.head(15)))

    path.write_text("\n".join(lines), encoding="utf-8")


def write_cases(path: Path, cases: list[FinalAuditCase]) -> None:
    path.write_text(
        json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoria final de prompts y flujo de produccion con 50 casos."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cases-only", action="store_true")
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Ejecuta solo extraccion/validacion, sin LightGBM/BERT.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=sorted(PROMPTS),
        default=list(PROMPTS),
        help="Subconjunto de prompts a ejecutar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "cases": out_dir / "audit_cases.json",
        "results_csv": out_dir / "audit_results.csv",
        "results_parquet": out_dir / "audit_results.parquet",
        "summary_csv": out_dir / "audit_summary.csv",
        "summary_json": out_dir / "audit_summary.json",
        "cache": out_dir / "audit_cache.jsonl",
        "report": out_dir / "audit_summary.md",
        "errors": out_dir / "audit_errors.md",
    }

    if args.force and paths["cache"].exists():
        paths["cache"].unlink()

    cases = construir_casos_finales()
    if len(cases) != 50:
        raise RuntimeError(f"La auditoria final debe contener 50 casos, hay {len(cases)}.")
    if args.limit is not None:
        cases = cases[: args.limit]
    write_cases(paths["cases"], cases)

    if args.cases_only:
        print(f"Casos guardados en: {paths['cases']}")
        return

    prompt_paths = {prompt_id: PROMPTS[prompt_id] for prompt_id in args.prompts}
    prompt_texts = {
        prompt_id: prompt_path.read_text(encoding="utf-8")
        for prompt_id, prompt_path in prompt_paths.items()
    }
    predictor = None if args.no_predict else TriajePredictor()
    cache = load_cache(paths["cache"])
    results: list[dict[str, Any]] = []

    total = len(cases) * len(prompt_paths)
    current = 0
    for prompt_id, prompt_path in prompt_paths.items():
        for case in cases:
            current += 1
            cache_key = f"{prompt_id}::{case.case_id}"
            if cache_key in cache:
                row = cache[cache_key]
                print(f"{current:03d}/{total} {prompt_id} {case.case_id} cache")
            else:
                print(f"{current:03d}/{total} {prompt_id} {case.case_id}")
                row = process_case(
                    case,
                    prompt_id=prompt_id,
                    prompt_path=prompt_path,
                    prompt_text=prompt_texts[prompt_id],
                    model=args.model,
                    predictor=predictor,
                )
                append_jsonl(paths["cache"], row)
            results.append(row)

    results_df = pd.DataFrame(results)
    summary_df = build_summary(results_df)
    results_df.to_csv(paths["results_csv"], index=False, encoding="utf-8")
    results_df.to_parquet(paths["results_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False, encoding="utf-8")
    paths["summary_json"].write_text(
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(paths["report"], summary_df, results_df)
    write_errors_report(paths["errors"], results_df)

    print(f"\nResultados guardados en: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

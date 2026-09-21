"""
Reference data for wound-graft / cellular-tissue-product (CTP) prior-authorization
appeals. Static tables only — no patient data.

HCPCS codes mirror PA_Biological_Dressing_Template.docx's existing "HCPCS Code Quick
Reference" table. ActiGraft's code is intentionally left as None (not guessed) since
it isn't independently verified — same convention the template already uses for the
generic "Other CTP" row.
"""
from __future__ import annotations

CTP_PRODUCTS: dict[str, dict] = {
    "apligraf": {
        "display": "Apligraf",
        "hcpcs": "Q4101",
        "category": "Bioengineered bilayered living skin construct",
        "indications": ["diabetic foot ulcer", "venous leg ulcer"],
    },
    "oasis": {
        "display": "OASIS",
        "hcpcs": "Q4103",
        "category": "Porcine small intestine submucosa matrix",
        "indications": ["diabetic foot ulcer", "venous leg ulcer"],
    },
    "dermagraft": {
        "display": "Dermagraft",
        "hcpcs": "Q4106",
        "category": "Bioengineered dermal substitute",
        "indications": ["diabetic foot ulcer"],
    },
    "epifix": {
        "display": "Epifix (MiMedx)",
        "hcpcs": "Q4116",
        "category": "Dehydrated human amniotic membrane allograft",
        "indications": ["diabetic foot ulcer", "venous leg ulcer"],
    },
    "strattice": {
        "display": "Strattice",
        "hcpcs": "Q4130",
        "category": "Acellular porcine dermal matrix",
        "indications": ["surgical wound", "abdominal wall reconstruction"],
    },
    "kerecis": {
        "display": "Kerecis",
        "hcpcs": "Q4158",
        "category": "Intact fish skin graft (Omega3 wound matrix)",
        "indications": ["diabetic foot ulcer", "burn"],
    },
    "amniofix": {
        "display": "Acell/Amniofix",
        "hcpcs": "Q4186",
        "category": "Amniotic/chorionic membrane allograft",
        "indications": ["diabetic foot ulcer", "venous leg ulcer"],
    },
    "actigraft": {
        "display": "ActiGraft",
        "hcpcs": None,
        "category": "Autologous whole blood clot matrix (point-of-care)",
        "indications": ["diabetic foot ulcer", "venous leg ulcer", "surgical wound"],
    },
    "other": {
        "display": "Other CTP",
        "hcpcs": "Q4199",
        "category": "Other cellular/tissue-based product",
        "indications": [],
    },
}

DENIAL_REASONS: dict[str, str] = {
    "not_medically_necessary":         "Not medically necessary",
    "investigational_experimental":    "Investigational / experimental",
    "conservative_care_not_exhausted": "Conservative/standard care not exhausted",
    "other":                           "Other / unspecified",
}

import json
from pypdf import PdfReader
from groq_client import call_llm


# ---------------------------
# PDF extractor
# ---------------------------

def extract_text_from_pdf_filelike(file_obj) -> str:
    """
    Extracts all text from a PDF uploaded via Streamlit (file-like object).
    """
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------------------------
# Agent 1 – Detect MOF/COF paper
# ---------------------------

def agent1_filter_and_detect(title: str, text_snippet: str) -> dict:
    """
    Decide whether a paper is about MOF/COF synthesis and extract basic info.
    """

    system = """
You are an expert in metal-organic frameworks (MOFs) and covalent-organic frameworks (COFs).
MOF/COF-related articles are likely to contain keywords like:
MOF, metal-organic framework, COF, covalent organic framework, organic–inorganic framework,
hybrid organic–inorganic materials, metal–organic polymer, coordination polymer, organic zeolite.

Given a paper title and text snippet, decide:
1) Is this paper reporting synthesis of a MOF or COF?
2) List any material names and their applications (if mentioned).

Reply ONLY in this JSON format:
{
  "is_mof_paper": true or false,
  "mof_names": ["..."],
  "applications": ["..."],
  "reason": "short explanation"
}
"""

    user = f"Title: {title}\n\nSnippet:\n{text_snippet[:3000]}"

    raw = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])


# ---------------------------
# Agent 2 – Extract detailed MOF/COF parameters
# ---------------------------

def agent2_extract_parameters(full_text: str) -> list:
    """
    Extracts detailed parameters from a COF/MOF synthesis article.
    Each returned entry corresponds to ONE material in the article.
    """

    schema = """
For EACH COF/MOF material described in the article, return one JSON object:

{
  "article_info": {
    "doi": "Article DOI or not_specified",
    "title": "Article title",
    "material_name": "Name of the COF/MOF material (e.g., N3-COF, MOF-801)"
  },

  "reactants": {
    "organic_linker_name": "Chemical name of organic linker/ligand or not_specified",
    "organic_linker_quantity_mg": 0,
    "metal_node_name": "Chemical name of metal core/node or not_specified",
    "metal_node_quantity_mg": 0,
    "solvent_name": "Solvent name(s) or not_specified",
    "solvent_quantity_ml": 0
  },

  "synthesis_conditions": {
    "reaction_time_seconds": 0,
    "reaction_temperature_celsius": 0,
    "stirring": "yes|no|not_specified",
    "total_reaction_time_seconds": 0,
    "stepwise_segmentation": 0,
    "ratio_components": "Molar ratio (e.g., 1:1:2) or not_specified",
    "annealing_time_seconds": 0,
    "annealing_temperature_celsius": 0
  },

  "morphology": {
    "pore_size_nm": 0,
    "pore_width_nm": 0,
    "pore_distribution_nm": "range in nm or not_specified"
  },

  "thermal_properties": {
    "breakdown_temperature_celsius": 0
  },

  "surface_properties": {
    "surface_area_m2_per_g": 0,
    "functional_groups": "List of functional groups or not_specified"
  },

  "chemical_properties": {
    "ph_range_min": 0,
    "ph_range_max": 0
  },

  "structure": {
    "nanocrystalline": "yes|no|not_specified",
    "amorphous": "yes|no|not_specified",
    "polar": "yes|no|not_specified",
    "nonpolar": "yes|no|not_specified",
    "dimensionality": "1D|2D|3D|not_specified"
  },

  "application": {
    "application": "Primary application or purpose (e.g., gas storage, catalysis, water harvesting)"
  }
}
"""

    system = f"""
You are an expert in COF/MOF chemistry and data extraction.

Extract the following parameters from the article text.

RULES:
- One JSON object per distinct COF/MOF material.
- Use the nested structure exactly as in the schema below: article_info, reactants,
  synthesis_conditions, morphology, thermal_properties, surface_properties,
  chemical_properties, structure, application.
- If a value is missing, use the string "not_specified" (except for numeric fields,
  where you may use 0 if truly no value is given).
- Numeric fields: output only the number, no units.
- Yes/no fields: use only "yes", "no", or "not_specified".
- Follow units: mg, ml, seconds, nm, °C.
- It is acceptable to approximate from the text if the value is clearly implied.

Return ONLY a JSON array (list) of objects matching this schema.

Schema:
{schema}
"""

    user = f"Article text (possibly truncated):\n{full_text[:12000]}"

    raw = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        return json.loads(raw[start:end])


# ---------------------------
# Helper – Summarise MOF/COF for application reasoning
# ---------------------------

def summarize_mof_for_application(mof_entry: dict) -> str:
    """
    Build a short natural-language summary of the MOF/COF from the extracted
    parameter schema to help Agent 3 reason more accurately.
    """
    ai = mof_entry.get("article_info", {})
    reac = mof_entry.get("reactants", {})
    cond = mof_entry.get("synthesis_conditions", {})
    morph = mof_entry.get("morphology", {})
    therm = mof_entry.get("thermal_properties", {})
    surf = mof_entry.get("surface_properties", {})
    chem = mof_entry.get("chemical_properties", {})
    struct = mof_entry.get("structure", {})
    appl = mof_entry.get("application", {})

    name = ai.get("material_name", "unnamed material")
    doi = ai.get("doi", "not_specified")

    linker = reac.get("organic_linker_name", "not_specified")
    metal = reac.get("metal_node_name", "not_specified")

    temp = cond.get("reaction_temperature_celsius", 0) or 0
    time_s = cond.get("reaction_time_seconds", 0) or 0
    steps = cond.get("stepwise_segmentation", 0) or 0

    sa = surf.get("surface_area_m2_per_g", 0) or 0
    tga = therm.get("breakdown_temperature_celsius", 0) or 0

    ph_min = chem.get("ph_range_min", 0) or 0
    ph_max = chem.get("ph_range_max", 0) or 0

    dim = struct.get("dimensionality", "not_specified")
    nano = struct.get("nanocrystalline", "not_specified")
    amorph = struct.get("amorphous", "not_specified")

    application = appl.get("application", "not_specified")

    summary_lines = [
        f"Material name: {name} (DOI: {doi}).",
        f"Metal node: {metal}; organic linker: {linker}.",
        f"Synthesis: temperature ~{temp} °C, time ~{time_s} s, stepwise segmentation = {steps}.",
        f"Surface area: {sa} m2/g (higher is better for adsorption).",
        f"Thermal stability (breakdown temperature): ~{tga} °C.",
        f"Reported pH stability range: {ph_min}–{ph_max}.",
        f"Structure: dimensionality = {dim}, nanocrystalline = {nano}, amorphous = {amorph}.",
        f"Reported primary application: {application}.",
    ]

    return " ".join(summary_lines)


# ---------------------------
# Agent 3 – Application suitability prediction (single score)
# ---------------------------

def agent3_predict_applications(mof_entry: dict) -> dict:
    """
    Given a single MOF/COF entry from Agent 2 (full parameter schema),
    infer likely application areas and assess overall suitability (0–100%)
    in a structured, evidence-based way.
    """
    import re as _re

    summary = summarize_mof_for_application(mof_entry)

    system = """
You are a senior MOF/COF applications expert.

Target context:
- We are particularly interested in **water treatment applications** (e.g., seawater, brine, wastewater),
  but you only need to output ONE overall suitability score per application (no separate Red Sea score).

You will receive:
1) A short natural-language summary of the material (synthesis, surface area, TGA, pH range, structure).
2) The full machine-extracted JSON parameters from a COF/MOF article.

First, internally estimate three key KPIs as percentages:
- stability_in_water_percent (0–100; 0 = very unstable, 100 = highly stable in water/seawater)
- synthetic_complexity_percent (0–100; 0 = very simple & mild synthesis, 100 = extremely complex/harsh)
- scalability_percent (0–100; 0 = not scalable, 100 = highly scalable & industrially friendly)

Use reaction temperature/time, number of steps, required conditions, TGA, pH range,
and any reported stability data to guide these values.

SCORING RULES (conceptual, must be followed):

For each candidate application, compute an overall suitability score out of 100% using the weights:

- Water stability:          25%
- Scalability:              20%
- Complexity (penalty):     20%
- Surface area:             20%
- Thermal stability (TGA):  15%

Define:
- surface_area_percent = min(surface_area_m2_per_g / 1500 * 100, 100)
- thermal_stability_percent = min(TGA_C / 500 * 100, 100)

First estimate a raw composite score:
- raw_score ≈
    0.25 * stability_in_water_percent
  + 0.20 * scalability_percent
  + 0.20 * (100 - synthetic_complexity_percent)
  + 0.20 * surface_area_percent
  + 0.15 * thermal_stability_percent

Then map it to the final reported suitability_score_percent as:
- suitability_score_percent ≈ round(0.6 * raw_score + 20), clipped to [0, 100].

This mapping is chosen so that:
- clearly poor materials are usually below ~40%,
- typical reasonable materials fall around 50–70%,
- exceptional candidates can reach 80–95%.

Your tasks:

1) Propose 3–5 plausible application scenarios for this material, chosen from or similar to:
   - pre-treatment of seawater / feedwater (reducing organics / fouling)
   - heavy metal removal from water
   - organic dye / pollutant adsorption from water
   - desalination brine treatment
   - gas storage / separation
   - catalysis
   - other clearly motivated use

2) For EACH scenario, evaluate:
   - suitability_score_percent: overall suitability of this MOF/COF for that application (0–100).
   - synthetic_complexity_percent (0–100; lower is better),
   - scalability_percent (0–100),
   - stability_in_water_percent (0–100),
   - surface_area_percent (0–100),
   - thermal_stability_percent (0–100),
   - key_supporting_properties: list of 3–6 short phrases directly referencing the material's properties
     (e.g., "BET surface area ~1200 m2/g", "TGA stable up to 450 °C", "pH-stable 3–11").
   - limitations: list of 2–4 short phrases highlighting uncertainties or weaknesses
     (e.g., "no explicit seawater stability data", "only moderate surface area", "narrow pH range").

3) Choose one application as **best_application** and justify it based on the properties.

Reply ONLY in valid JSON with this structure:

{
  "application_candidates": [
    {
      "name": "short name of application",
      "category": "water_treatment | gas_storage | catalysis | other",
      "suitability_score_percent": 0,
      "synthetic_complexity_percent": 0,
      "scalability_percent": 0,
      "stability_in_water_percent": 0,
      "surface_area_percent": 0,
      "thermal_stability_percent": 0,
      "key_supporting_properties": ["..."],
      "limitations": ["..."]
    }
  ],
  "best_application": {
    "name": "short name of best application",
    "reason": "3–6 sentences explicitly linking the application with the material's properties."
  },
  "uncertainties": "2–5 sentences summarising what is unknown or poorly supported in the article."
}

Be conservative. If important data (e.g., water stability) is missing, mention this in 'limitations' and 'uncertainties'.
"""

    user = (
        "COF/MOF summary:\n"
        f"{summary}\n\n"
        "Full parameter JSON from Agent 2:\n"
        f"{json.dumps(mof_entry, indent=2)}"
    )

    raw = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    def try_parse(s: str):
        s = s.strip()
        if s.startswith("```"):
            s = _re.sub(r"^```[a-zA-Z]*\n", "", s)
            if s.endswith("```"):
                s = s[:-3]
        return json.loads(s)

    try:
        return try_parse(raw)
    except Exception:
        try:
            m = _re.search(r"\{.*\}", raw, _re.S)
            if m:
                return try_parse(m.group(0))
        except Exception as e2:
            return {
                "parse_error": f"Could not parse Agent 3 JSON: {e2}",
                "raw_response": raw,
            }

        return {
            "parse_error": "No JSON object found in Agent 3 response.",
            "raw_response": raw,
        }

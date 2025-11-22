import streamlit as st
import pandas as pd
from io import BytesIO

from agents import (
    extract_text_from_pdf_filelike,
    agent1_filter_and_detect,
    agent2_extract_parameters,
    agent3_predict_applications,
)

st.set_page_config(page_title="GenMOF – Groq-powered MOF App", layout="wide")

st.title("🧪 GenMOF – Groq-powered MOF/COF Synthesis & Application Explorer")

st.markdown(
    """
This prototype uses **Groq-hosted LLMs** to help with **COF/MOF discovery**
for **water treatment applications**.

Pipeline:
1. **Agent 1** – Detects whether the paper reports MOF/COF synthesis.
2. **Agent 2** – Extracts structured synthesis parameters into a rich schema.
3. **Agent 3** – Predicts likely applications and an overall suitability score (%)
   based on stability, complexity, scalability, surface area, and thermal robustness.
"""
)

uploaded_files = st.file_uploader(
    "Upload COF/MOF PDF articles", type=["pdf"], accept_multiple_files=True
)

if st.button("🚀 Run Pipeline") and uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"📄 {file.name}")

        # 1) Extract full text
        with st.spinner("Extracting PDF text..."):
            text = extract_text_from_pdf_filelike(file)

        article_word_count = len(text.split())
        snippet = text[:2000]

        # 2) Agent 1 – MOF/COF detection
        with st.spinner("Agent 1: Detecting if this is a MOF/COF synthesis paper..."):
            a1 = agent1_filter_and_detect(file.name, snippet)

        a1["article_word_count"] = article_word_count

        st.markdown("### 🔍 Agent 1 – MOF/COF Detection")
        st.json(a1)

        if not a1.get("is_mof_paper", False):
            st.warning("❌ Not identified as a MOF/COF synthesis paper. Skipping further analysis.")
            continue

        # 3) Agent 2 – Detailed parameter extraction
        with st.spinner("Agent 2: Extracting detailed material parameters..."):
            entries = agent2_extract_parameters(text)

        st.markdown("### ⚗️ Agent 2 – Full Extracted Parameter Set (per material)")
        st.json(entries)

        if not entries:
            st.warning("No COF/MOF materials extracted by Agent 2.")
            continue

        # ------- Flatten entries into a DataFrame -------
        def flatten_entry(entry: dict) -> dict:
            flat = {}
            for section_name, section_value in entry.items():
                if isinstance(section_value, dict):
                    for k, v in section_value.items():
                        flat[f"{section_name}_{k}"] = v
                else:
                    flat[section_name] = section_value
            return flat

        flat_entries = [flatten_entry(e) for e in entries]
        df = pd.DataFrame(flat_entries)

        # Pretty column names
        rename_map = {
            "article_info_doi": "DOI",
            "article_info_title": "Title",
            "article_info_material_name": "Material",
            "reactants_organic_linker_name": "Organic linker",
            "reactants_metal_node_name": "Metal node",
            "synthesis_conditions_reaction_temperature_celsius": "Reaction temp (°C)",
            "synthesis_conditions_reaction_time_seconds": "Reaction time (s)",
            "surface_properties_surface_area_m2_per_g": "Surface area (m2/g)",
            "thermal_properties_breakdown_temperature_celsius": "TGA breakdown (°C)",
            "chemical_properties_ph_range_min": "pH min",
            "chemical_properties_ph_range_max": "pH max",
            "application_application": "Application",
        }
        df.rename(columns=rename_map, inplace=True)

        must_have_cols = [
            "DOI",
            "Title",
            "Material",
            "Organic linker",
            "Metal node",
            "Reaction temp (°C)",
            "Reaction time (s)",
            "Surface area (m2/g)",
            "TGA breakdown (°C)",
            "Application",
        ]
        display_cols = [c for c in must_have_cols if c in df.columns]

        st.markdown("#### ⭐ Key Parameters (must-have subset)")
        if display_cols:
            st.dataframe(df[display_cols])
        else:
            st.info("Key parameters not available for this article; showing full JSON above.")

        # ------- Excel download with ALL parameters -------
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="COF_MOF_parameters")
        excel_data = output.getvalue()

        st.download_button(
            label="💾 Download all parameters as Excel",
            data=excel_data,
            file_name=f"cof_mof_parameters_{file.name.replace('.pdf','')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # ------- Choose a "best" material for Agent 3 -------
        best_index = 0
        if "Surface area (m2/g)" in df.columns:
            surf = pd.to_numeric(df["Surface area (m2/g)"], errors="coerce").fillna(0)
            if not surf.empty:
                best_index = int(surf.idxmax())

        best_entry = entries[best_index]

        # 4) Agent 3 – Application prediction & suitability
        with st.spinner("Agent 3: Predicting applications & suitability..."):
            app_predictions = agent3_predict_applications(best_entry)

        st.markdown("### 🔮 Agent 3 – Predicted Applications & Suitability")
        st.json(app_predictions)

        # Compact scoring table (overall suitability only)
        if isinstance(app_predictions, dict) and "application_candidates" in app_predictions:
            try:
                apps_df = pd.DataFrame(app_predictions["application_candidates"])
                if not apps_df.empty:
                    cols = [
                        "name",
                        "category",
                        "suitability_score_percent",
                    ]
                    cols = [c for c in cols if c in apps_df.columns]
                    if cols:
                        apps_df = apps_df[cols]
                        pretty_cols = {
                            "name": "Application",
                            "category": "Category",
                            "suitability_score_percent": "Suitability (%)",
                        }
                        apps_df.rename(columns=pretty_cols, inplace=True)

                        st.markdown("#### 📊 Application scoring overview")
                        st.dataframe(apps_df)
            except Exception:
                pass

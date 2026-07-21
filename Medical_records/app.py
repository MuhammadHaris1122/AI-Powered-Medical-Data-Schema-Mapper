"""
================================================================================
AI-Powered Medical Data Schema Mapper
================================================================================
A Streamlit application that automatically maps a source medical CSV to the
schema of a target medical CSV using AI/NLP techniques (Sentence Transformer
embeddings, cosine similarity, medical synonym matching, RapidFuzz fuzzy
matching, and data-type/ID validation).

NOTE: All AI/ML logic below is taken directly from the original Jupyter
Notebook (AI-schema-Mapper.ipynb) and has NOT been altered. This file only
wraps that logic inside a Streamlit user interface.

Run with:
    streamlit run app.py
================================================================================
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI Medical Schema Mapper",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# (Keeps data persistent across button clicks / reruns)
# ==============================================================================
for key in [
    "source_df", "target_df", "mapping_df", "similarity_df",
    "transformed_df", "column_mapping"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ==============================================================================
# CACHED MODEL LOADER
# (Loading the Sentence Transformer model is expensive, so we cache it)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pre-trained Sentence Transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# ORIGINAL NOTEBOOK LOGIC
# (Unchanged AI / matching / transformation functions from the notebook)
# ==============================================================================

def analyze_schema(df):
    """Analyze the schema of a dataset (column name, dtype, missing, unique)."""
    schema_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    return schema_info


def generate_metadata(df):
    """Generate metadata (name, dtype, missing, unique, sample values) per column."""
    metadata = []
    for column in df.columns:
        metadata.append({
            "Column Name": column,
            "Data Type": str(df[column].dtype),
            "Missing Values": df[column].isnull().sum(),
            "Unique Values": df[column].nunique(),
            "Sample Values": list(df[column].dropna().unique()[:5])
        })
    return pd.DataFrame(metadata)


def is_id_column(column_name):
    """Check whether a column is an ID column based on naming keywords."""
    column_name = column_name.lower()
    id_keywords = [
        "id", "record_id", "patient_id", "emp_id", "employee_id", "customer_id"
    ]
    return any(keyword in column_name for keyword in id_keywords)


def datatype_similarity(source_dtype, target_dtype):
    """Return 1.0 if datatypes match exactly, else 0.0."""
    if source_dtype == target_dtype:
        return 1.0
    return 0.0


# Medical synonym dictionary used for domain-aware column matching
medical_synonyms = {
    "blood sugar": ["glucose", "blood_glucose"],
    "bp": ["bloodpressure", "blood_pressure"],
    "sex": ["gender"],
    "body mass index": ["bmi"],
    "disease": ["diagnosis"],
    "patient age": ["age"],
    "cholesterol level": ["cholesterol"],
    "serum insulin": ["insulin"],
    "record_id": ["patient_id"]
}


def synonym_similarity(source_col, target_col):
    """Return 1 if source and target column names are known medical synonyms."""
    source = source_col.lower().strip()
    target = target_col.lower().strip()

    if source == target:
        return 1

    if source in medical_synonyms:
        if target in medical_synonyms[source]:
            return 1

    return 0


def exact_match(source_col, target_col):
    """Return True if both column names are exactly the same (case-insensitive)."""
    return source_col.lower().strip() == target_col.lower().strip()


def fuzzy_similarity(source_col, target_col):
    """Calculate RapidFuzz token-sort-ratio based fuzzy similarity (0-1)."""
    score = fuzz.token_sort_ratio(source_col, target_col)
    return score / 100


def hybrid_similarity(semantic_score, source_col, target_col, source_dtype, target_dtype):
    """
    Multi-Level Matching Score.
    Combines: ID rule, exact match, semantic similarity, datatype bonus,
    medical synonym bonus, and fuzzy string bonus.
    """
    # Rule 1: ID columns must match on both sides or neither
    if is_id_column(source_col) != is_id_column(target_col):
        return 0

    # Rule 2: Exact Match
    if exact_match(source_col, target_col):
        return 1.0

    # Rule 3: Semantic Score (base)
    score = semantic_score

    # Rule 4: Datatype Bonus
    if datatype_similarity(source_dtype, target_dtype):
        score += 0.10

    # Rule 5: Synonym Bonus
    if synonym_similarity(source_col, target_col):
        score += 0.15

    # Rule 6: Fuzzy Bonus
    fuzzy_score = fuzzy_similarity(source_col, target_col)
    if fuzzy_score >= 0.80:
        score += 0.10
    elif fuzzy_score >= 0.60:
        score += 0.05

    # Final score cannot exceed 1
    score = min(score, 1.0)

    return round(score, 4)


# Standardization rules used during the transform step
standardization_rules = {
    "Gender": {"M": "Male", "F": "Female", "Male": "Male", "Female": "Female"},
    "Smoker": {"Yes": 1, "No": 0, "Y": 1, "N": 0}
}


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 🩺 AI Medical Schema Mapper")
    st.markdown("---")

    st.markdown("### 📌 About Project")
    st.info(
        "This application automatically converts a **source medical CSV** "
        "into the schema of a **target medical CSV**, without any hardcoded "
        "column mappings — powered entirely by AI/NLP techniques."
    )

    st.markdown("### 🛠️ Technologies Used")
    st.markdown(
        """
        - **Streamlit** – Web application framework
        - **Sentence Transformers** – Semantic embeddings (`all-MiniLM-L6-v2`)
        - **Scikit-learn** – Cosine similarity
        - **RapidFuzz** – Fuzzy string matching
        - **Pandas / NumPy** – Data processing
        """
    )

    st.markdown("### 👨‍💻 Developer Information")
    st.markdown(
        """
        **Developed by:** Haris Ch
        **Project Type:** AI & Machine Learning Internship Project
        """
    )

    st.markdown("---")
    st.caption("Upload your CSV files to get started ⬆️")


# ==============================================================================
# HEADER
# ==============================================================================
st.title("🩺 AI-Powered Medical Data Schema Mapper")
st.markdown(
    "Automatically map and transform a **source** medical dataset into the "
    "schema of a **target** medical dataset using AI-based column matching."
)
st.markdown("---")


# ==============================================================================
# STEP 1: FILE UPLOAD
# ==============================================================================
st.header("📁 Step 1: Upload Datasets")

col_upload_1, col_upload_2 = st.columns(2)

with col_upload_1:
    source_file = st.file_uploader("Upload Source CSV", type=["csv"], key="source_upload")

with col_upload_2:
    target_file = st.file_uploader("Upload Target CSV", type=["csv"], key="target_upload")

if source_file is not None:
    st.session_state.source_df = pd.read_csv(source_file)
    st.success("✅ Source CSV uploaded successfully!")

if target_file is not None:
    st.session_state.target_df = pd.read_csv(target_file)
    st.success("✅ Target CSV uploaded successfully!")

st.markdown("---")


# ==============================================================================
# STEP 2: DATA PREVIEW
# ==============================================================================
if st.session_state.source_df is not None and st.session_state.target_df is not None:

    source_df = st.session_state.source_df
    target_df = st.session_state.target_df

    st.header("👀 Step 2: Data Preview")

    preview_col_1, preview_col_2 = st.columns(2)

    with preview_col_1:
        st.subheader("Source Dataset")
        st.dataframe(source_df.head(), use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric("Rows", source_df.shape[0])
        m2.metric("Columns", source_df.shape[1])
        with st.expander("📋 Source Column Names"):
            st.write(list(source_df.columns))

    with preview_col_2:
        st.subheader("Target Dataset")
        st.dataframe(target_df.head(), use_container_width=True)
        m3, m4 = st.columns(2)
        m3.metric("Rows", target_df.shape[0])
        m4.metric("Columns", target_df.shape[1])
        with st.expander("📋 Target Column Names"):
            st.write(list(target_df.columns))

    st.markdown("---")

    # ==========================================================================
    # STEP 3: AI MAPPING
    # ==========================================================================
    st.header("🤖 Step 3: AI-Powered Column Mapping")

    if st.button("🚀 Generate Mapping", type="primary"):

        progress_bar = st.progress(0, text="Starting AI mapping pipeline...")

        with st.spinner("Analyzing schema and generating metadata..."):
            source_metadata = generate_metadata(source_df)
            target_metadata = generate_metadata(target_df)
            progress_bar.progress(20, text="Schema analysis complete...")

        with st.spinner("Loading Sentence Transformer model..."):
            model = load_model()
            progress_bar.progress(40, text="Model loaded. Generating embeddings...")

        source_columns = source_df.columns.tolist()
        target_columns = target_df.columns.tolist()

        with st.spinner("Generating column embeddings..."):
            source_embeddings = model.encode(source_columns)
            target_embeddings = model.encode(target_columns)
            progress_bar.progress(60, text="Calculating cosine similarity...")

        # Cosine similarity matrix
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

        similarity_df = pd.DataFrame(
            similarity_matrix, index=source_columns, columns=target_columns
        )

        progress_bar.progress(75, text="Applying hybrid matching (synonyms + fuzzy + dtype)...")

        # Build hybrid similarity matrix using multi-level matching engine
        hybrid_matrix = similarity_matrix.copy()
        for i, source_col in enumerate(source_columns):
            for j, target_col in enumerate(target_columns):
                hybrid_matrix[i][j] = hybrid_similarity(
                    semantic_score=similarity_matrix[i][j],
                    source_col=source_col,
                    target_col=target_col,
                    source_dtype=str(source_df[source_col].dtype),
                    target_dtype=str(target_df[target_col].dtype)
                )

        progress_bar.progress(90, text="Generating final one-to-one mapping...")

        # Generate final one-to-one mapping (greedy, highest score first)
        mapping = []
        used_targets = set()

        for i, source_col in enumerate(source_columns):
            scores = hybrid_matrix[i].copy()
            while True:
                best_index = scores.argmax()
                target_col = target_columns[best_index]
                if target_col not in used_targets:
                    mapping.append({
                        "Source Column": source_col,
                        "Target Column": target_col,
                        "Similarity Score": round(scores[best_index], 4)
                    })
                    used_targets.add(target_col)
                    break
                scores[best_index] = -1

        mapping_df = pd.DataFrame(mapping)
        column_mapping = dict(zip(mapping_df["Source Column"], mapping_df["Target Column"]))

        # Persist results in session state
        st.session_state.mapping_df = mapping_df
        st.session_state.similarity_df = similarity_df
        st.session_state.column_mapping = column_mapping

        progress_bar.progress(100, text="Mapping complete!")
        st.success("✅ AI mapping generated successfully!")

    # Display mapping results if available
    if st.session_state.mapping_df is not None:

        st.subheader("🔗 Column Mapping Table")
        st.dataframe(st.session_state.mapping_df, use_container_width=True)

        avg_score = st.session_state.mapping_df["Similarity Score"].mean()
        min_score = st.session_state.mapping_df["Similarity Score"].min()

        s1, s2, s3 = st.columns(3)
        s1.metric("Average Similarity", f"{avg_score:.2%}")
        s2.metric("Lowest Match Score", f"{min_score:.2%}")
        s3.metric("Columns Mapped", len(st.session_state.mapping_df))

        if min_score < 0.5:
            st.warning(
                "⚠️ Some columns have a low similarity score (< 50%). "
                "Please review the mapping table carefully."
            )
        else:
            st.info("ℹ️ All mapped columns have a reasonably high similarity score.")

        with st.expander("📊 View Full Similarity Matrix (Cosine Similarity)"):
            st.dataframe(st.session_state.similarity_df, use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # STEP 4: TRANSFORMATION
    # ==========================================================================
    st.header("🔄 Step 4: Transform Dataset")

    if st.session_state.mapping_df is None:
        st.info("ℹ️ Please generate the column mapping first (Step 3).")
    else:
        if st.button("⚙️ Transform Dataset", type="primary"):

            with st.spinner("Transforming source dataset to target schema..."):

                column_mapping = st.session_state.column_mapping

                # Rename columns according to AI-generated mapping
                transformed_df = source_df.rename(columns=column_mapping)

                # Reorder columns to match target schema
                transformed_df = transformed_df[target_df.columns]

                # Apply value standardization (e.g., M -> Male, Yes -> 1)
                for column, value_map in standardization_rules.items():
                    if column in transformed_df.columns:
                        transformed_df[column] = transformed_df[column].replace(value_map)

                st.session_state.transformed_df = transformed_df

            st.success("✅ Dataset transformed successfully!")

        if st.session_state.transformed_df is not None:
            transformed_df = st.session_state.transformed_df

            st.subheader("📄 Transformed Dataset Preview")
            st.dataframe(transformed_df.head(10), use_container_width=True)

            t1, t2 = st.columns(2)
            t1.metric("Rows", transformed_df.shape[0])
            t2.metric("Columns", transformed_df.shape[1])

            with st.expander("📋 Transformed Column Names"):
                st.write(list(transformed_df.columns))

    st.markdown("---")

    # ==========================================================================
    # STEP 5: VALIDATION
    # ==========================================================================
    if st.session_state.transformed_df is not None:

        st.header("✅ Step 5: Validate Output")

        transformed_df = st.session_state.transformed_df

        # --- Column name / order validation ---
        st.subheader("🧩 Column Validation")
        if list(transformed_df.columns) == list(target_df.columns):
            st.success("✅ Column names and order match the target schema.")
        else:
            st.warning("⚠️ Column names or order do not fully match the target schema.")

        # --- Missing values ---
        st.subheader("🕳️ Missing Values Check")
        missing_report = pd.DataFrame({
            "Column": transformed_df.columns,
            "Missing Values": transformed_df.isnull().sum().values
        })
        st.dataframe(missing_report, use_container_width=True)

        total_missing = transformed_df.isnull().sum().sum()
        if total_missing == 0:
            st.success("✅ No missing values found in the transformed dataset.")
        else:
            st.warning(f"⚠️ Found {total_missing} missing values in the transformed dataset.")

        # --- Data type validation ---
        st.subheader("🔠 Data Type Validation")
        validation_df = pd.DataFrame({
            "Column": target_df.columns,
            "Target Data Type": target_df.dtypes.values,
            "Converted Data Type": transformed_df.dtypes.values
        })
        validation_df["Match"] = (
            validation_df["Target Data Type"].astype(str)
            == validation_df["Converted Data Type"].astype(str)
        )
        st.dataframe(validation_df, use_container_width=True)

        if validation_df["Match"].all():
            st.success("✅ All data types match the target schema.")
        else:
            st.warning("⚠️ Some data types do not match the target schema.")

        st.markdown("---")

        # ======================================================================
        # STEP 6: DOWNLOADS
        # ======================================================================
        st.header("⬇️ Step 6: Download Results")

        d1, d2 = st.columns(2)

        with d1:
            csv_buffer = io.StringIO()
            transformed_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Converted CSV",
                data=csv_buffer.getvalue(),
                file_name="converted_medical_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d2:
            if st.session_state.mapping_df is not None:
                mapping_csv_buffer = io.StringIO()
                st.session_state.mapping_df.to_csv(mapping_csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Mapping Report",
                    data=mapping_csv_buffer.getvalue(),
                    file_name="column_mapping_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

else:
    st.info("⬆️ Please upload both the Source CSV and Target CSV to begin.")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Developed by <b> Muhammad Haris</b></p>
        <p>AI & Machine Learning Internship Project</p>
    </div>
    """,
    unsafe_allow_html=True
)

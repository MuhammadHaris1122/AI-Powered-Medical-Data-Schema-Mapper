# AI-Powered Medical Data Schema Mapper

## Overview

The AI-Powered Medical Data Schema Mapper is a Machine Learning and NLP-based project that automatically maps a source medical dataset to a target schema without using hardcoded column mappings.

The project uses semantic similarity with Sentence Transformers, cosine similarity, RapidFuzz, and rule-based validation to identify the best matching columns between two medical datasets and generate a transformed dataset.

---

## Features

- Automatic medical schema mapping
- Semantic column matching using Sentence Transformers
- Cosine Similarity for column comparison
- RapidFuzz for fuzzy string matching
- Medical synonym matching
- Data type validation
- ID column detection
- Automatic dataset transformation
- CSV export
- Streamlit-based user interface

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Sentence Transformers
- RapidFuzz
- Streamlit
- OpenPyXL
- Matplotlib

---

## Project Workflow

1. Load Source Dataset
2. Load Target Dataset
3. Analyze Dataset Schema
4. Generate Metadata
5. Create Sentence Embeddings
6. Calculate Semantic Similarity
7. Generate Column Mapping
8. Apply Hybrid Matching Rules
9. Transform Source Dataset
10. Standardize Data
11. Validate Final Dataset
12. Export Converted CSV

---

## Project Structure

```text
AI-Medical-Data-Schema-Mapper/
│
├── data/
│   ├── source.csv
│   └── target.csv
│
├── output/
│   ├── Final_Medical_Dataset.csv
│   └── column_mapping_report.csv
│
├── notebooks/
│   └── AI_Medical_Schema_Mapper.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Medical-Data-Schema-Mapper.git
```

Move into the project directory:

```bash
cd AI-Medical-Data-Schema-Mapper
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
AI_Medical_Schema_Mapper.ipynb
```

### Streamlit Application

```bash
streamlit run app.py
```

---

## Expected Output

The project generates:

- Converted Medical Dataset (CSV)
- Column Mapping Report
- Validation Results
- Streamlit Web Interface

---

## Future Improvements

- Support for Excel files
- Automatic unit conversion
- Confidence score visualization
- Multiple dataset mapping
- Enhanced medical terminology support

---

## Author

**Haris Ch**

Computer Science Graduate

AI & Machine Learning Enthusiast

---

## License

This project is developed for educational and internship purposes.

import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Step 1: Load Data ===
diagnosis_df = pd.read_csv("Diagnoses_list.csv")  # Column: Diagnoses_list (stringified list)
icd_df = pd.read_csv("icd10_codes.csv")  # Columns: ICDCode, Description

# Clean column names
icd_df.columns = [col.strip() for col in icd_df.columns]
icd_df['Description'] = icd_df['Description'].str.strip()

# Convert stringified list to actual Python list
if diagnosis_df["Diagnoses_list"].dtype == object:
    diagnosis_df["Diagnoses_list"] = diagnosis_df["Diagnoses_list"].apply(ast.literal_eval)

# === Step 2: Load the sentence embedding model ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute ICD description embeddings
icd_descriptions = icd_df["Description"].tolist()
icd_embeddings = embedding_model.encode(icd_descriptions, convert_to_tensor=False)

# === Step 3: Function to map diagnosis to ICD code ===
def map_diagnosis_to_icd(diagnosis):
    diag_embedding = embedding_model.encode([diagnosis])
    sims = cosine_similarity(diag_embedding, icd_embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    return {
        "diagnosis": diagnosis,
        "code": icd_df.iloc[best_idx]["ICDCode"],
        "description": icd_df.iloc[best_idx]["Description"],
        "score": best_score
    }

# === Step 4: Process all rows and save output ===
results = []
for row_id, row in diagnosis_df.iterrows():
    for diag in row["Diagnoses_list"]:
        mapped = map_diagnosis_to_icd(diag)
        results.append({
            "Row_ID": row_id,  # <-- Add this line
            "Original Diagnosis": mapped["diagnosis"],
            "ICD-10 Code": mapped["code"],
            "ICD-10 Description": mapped["description"],
            "Similarity Score": round(mapped["score"], 3)
        })

output_df = pd.DataFrame(results)
output_df.to_csv("output.csv", index=False)
print("Saved output to output.csv") 
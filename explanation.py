import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
from tqdm import tqdm
import re

# --- CONFIG ---
SIMILARITY_THRESHOLD = 0.75  # Not used now, but kept for reference
LOCAL_MODEL_NAME = "bkholyday/Qwen2.5-0.5B-Instruct-medicalLLM-HuatuoGPT-o1-sft"  # Medical LLM, or use BioGPT-Large if you have resources
BATCH_SIZE = 8  # Adjust based on your hardware
CHECKPOINT_EVERY = 50
INPUT_FILE = "output_chunks/output_part_2.csv"  # Set your chunk file here
OUTPUT_DIR = "flot"  # Save results here
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE).replace(".csv", "_rag.csv"))
PARTIAL_FILE = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE).replace(".csv", "_rag_partial.csv"))

# --- SETUP ---
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=60,       # Enforces short output
    do_sample=True,          # Enables variation
    temperature=0.3,         # Lower = more focused
    top_p=0.9,               # Limits output to top 90% probs
    batch_size=BATCH_SIZE,
    return_full_text=False
)

def build_rag_prompt(diagnosis, code, description, score):
    # return (
    #     "You are a board‑certified medical coder. "
    #     "Given a patient’s diagnosis and its assigned ICD‑10 code, output exactly TWO complete sentences explaining why the code is correct. "
    #     "Do NOT include IDs, unfinished phrases, or meta‑comments—only clinical reasoning.\n\n"
    #     f"Diagnosis: “{diagnosis}”\n"
    #     f"Assigned ICD‑10 Code: {code} — “{description}”\n"
    #     f"Similarity Score: {score:.2f}\n\n"
    #     "Explanation:"
    # )
    return (
        "You are a medical coding expert. Given a clinical diagnosis and its matching ICD-10 code, explain briefly and precisely in **exactly two clinical sentences** why the code is appropriate. Use medical reasoning. End the explanation with '<END>'.\n\n"
        "### Example 1\n"
        "Diagnosis: Pure hypercholesterolemia\n"
        "ICD-10 Code: E7800 — Pure hypercholesterolemia\n"
        "Explanation: The ICD-10 code E7800 correctly identifies isolated elevation of cholesterol without other lipid abnormalities. It captures pure hypercholesterolemia in the absence of secondary causes or comorbid conditions. <END>\n\n"
        "### Example 2\n"
        "Diagnosis: Unspecified acquired hypothyroidism\n"
        "ICD-10 Code: E039 — Hypothyroidism, unspecified\n"
        "Explanation: E039 accurately reflects the patient's hypothyroid symptoms in the absence of a clearly identified etiology. It provides appropriate classification when lab findings confirm hypothyroidism but the cause remains undetermined. <END>\n\n"
        "### Now you:\n"
        f"Diagnosis: {diagnosis}\n"
        f"ICD-10 Code: {code} — {description}\n"
        f"Similarity Score: {score:.2f}\n"
        "Explanation:"
    )





def generate_explanations_local(prompts):
    results = generator(prompts)
    explanations = []
    # Flatten if results is a list of lists
    if results and isinstance(results[0], list):
        results = [item for sublist in results for item in sublist]
    for prompt, result in zip(prompts, results):
        text = result['generated_text'].replace(prompt, '').strip()
        if len(text) < 20 or not any(c.isalpha() for c in text):
            text = "This ICD-10 code was selected based on semantic similarity to the diagnosis and description."
        explanations.append(text)
    return explanations

def clean_explanation(text):
    import re

    # Remove redundant phrases
    text = re.sub(r'The patient has a diagnosis of |This can be caused by.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'This information can be used to.*?\.', '', text, flags=re.IGNORECASE)

    # Fix common double punctuation errors
    text = re.sub(r'\.\s*\.', '.', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure ends with <END>
    if not text.endswith("<END>"):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 2:
            text = sentences[0] + ' ' + sentences[1]
        else:
            text = sentences[0]
        if not text.endswith('.'):
            text += '.'
        text += ' <END>'
    else:
        text = text.replace(' <END><END>', ' <END>')

    return text



def main():
    df = pd.read_csv(INPUT_FILE)
    if os.path.exists(PARTIAL_FILE):
        done_df = pd.read_csv(PARTIAL_FILE)
        done_indices = set(done_df.index)
        print(f"Resuming: {len(done_indices)} rows already processed.")
    else:
        done_df = pd.DataFrame()
        done_indices = set()

    output_rows = []
    batch_prompts = []
    batch_rows = []
    for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if i in done_indices:
            continue
        try:
            prompt = build_rag_prompt(row["Original Diagnosis"], row["ICD-10 Code"], row["ICD-10 Description"], row["Similarity Score"])
            explanation = clean_explanation(generate_explanations_local([prompt])[0].strip())
            model_used = "Local LLM"
            output_row = row.to_dict()
            output_row["RAG Explanation"] = explanation
            output_row["Model Used"] = model_used
            print("OUTPUT_ROW TYPE:", type(output_row))
            print("OUTPUT_ROW CONTENT:", output_row)
            output_rows.append(output_row)
        except Exception as e:
            print("ERROR AT ROW", i)
            print("ROW TYPE:", type(row))
            print("ROW CONTENT:", row)
            print("EXCEPTION:", e)
            output_rows.append({"RAG Explanation": f"ERROR: {e}"})
            break
        # Checkpoint after each row
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(output_rows).to_csv(PARTIAL_FILE, index=False)
    # Save final
    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
# ICD-10 Code Mapping AI System 

## Overview
This project is part of an assignment ([see ass.md](input_files/ass.md)) to develop an AI-based system that maps patient diagnoses to ICD-10 codes, and generates clinical explanations for each mapping. The workflow is designed for accuracy, scalability, and extensibility.

## Data
- **Input:** `Diagnoses_list.csv` (in `input_files/`)
  - Each row contains a Python-list string of diagnoses for a single patient.
- **ICD-10 Metadata:** Publicly available ICD-10 code descriptions (see links in `ass.md`).

## Workflow: Step-by-Step

### 1. ICD-10 Code Embedding & Mapping (`embed_and_map_icd.py`)
- **Model Used:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Process:**
  1. **Load Diagnoses:** Each row in `Diagnoses_list.csv` is parsed into a list of diagnoses per patient.
  2. **Flatten:** Each diagnosis is split into its own row, with a `Row_ID` to track the patient.
  3. **ICD-10 Embedding:** All ICD-10 code descriptions are embedded as vectors using the MiniLM model.
  4. **Diagnosis Embedding:** Each diagnosis is embedded as a vector.
  5. **Cosine Similarity:** For each diagnosis, cosine similarity is computed against all ICD-10 code vectors.
  6. **Best Match:** The ICD-10 code with the highest similarity is selected for each diagnosis, along with its description and similarity score.
  7. **Output:** Results are saved to `output.csv` (columns: Row_ID, Original Diagnosis, ICD-10 Code, ICD-10 Description, Similarity Score).

### 2. Chunking for LLM Processing
- **Rationale:** LLMs are more efficient and robust when processing smaller batches. Chunking also enables parallel processing and easier checkpointing.
- **Process:**
  - `output.csv` is split into 500-row chunks and stored in `output_chunks/`.
  - This allows for distributed or incremental processing, and makes recovery from interruptions straightforward.

### 3. Clinical Explanation Generation (`explanation.py`)
- **Model Used:** [`bkholyday/Qwen2.5-0.5B-Instruct-medicalLLM-HuatuoGPT-o1-sft`](https://huggingface.co/bkholyday/Qwen2.5-0.5B-Instruct-medicalLLM-HuatuoGPT-o1-sft) (local LLM)
- **Prompt Engineering:**
  - The prompt is designed to elicit exactly two clinical sentences justifying the ICD-10 code for each diagnosis, using only the diagnosis and code description.
  - Example-based prompting is used to guide the model's style and content.
- **Process:**
  1. Each chunk is loaded and processed row by row.
  2. For each diagnosis-code pair, a prompt is constructed and sent to the LLM.
  3. The LLM generates a concise, clinical explanation, which is post-processed for consistency and brevity.
  4. **Checkpointing:** After every N rows (default 50), progress is saved to a partial file. This ensures that if the process is interrupted, it can resume from the last checkpoint without reprocessing completed rows.
  5. **Output:** Explanations are saved in `flot/` as `output_part_X_rag.csv`.

### 4. Aggregation and Patient-Level Output
- **Process:**
  - All chunked explanation files are concatenated.
  - For each patient (`Row_ID`), all diagnoses, codes, descriptions, similarity scores, and explanations are grouped as Python lists.
  - This produces a single row per patient, with all relevant information for downstream analysis or reporting.
  - **Output:** `diag_code_exp.csv` (one row per patient, with all relevant info as lists).

## Data Flow and Robustness
- **Intermediate CSVs** are used at each stage for transparency, debugging, and model selection.
- **Checkpointing** ensures that long-running LLM jobs are robust to crashes or interruptions.
- **Chunking** enables parallel or distributed processing, and makes it easy to scale to large datasets.

## Model Selection and Extensions
- **Embedding Model:** `all-MiniLM-L6-v2` is chosen for its speed and strong semantic matching performance.
- **LLM for Explanations:** `Qwen2.5-0.5B-Instruct-medicalLLM-HuatuoGPT-o1-sft` is used for its medical instruction tuning and efficiency on local hardware.
- **Possible Extensions:**
  - Swap in larger or more specialized LLMs for higher-quality explanations.
  - Fine-tune the LLM on your own explanation data for even better results.
  - Use other chunk sizes or batch strategies for different hardware.
  - Integrate with LangChain or other frameworks for more complex RAG workflows.

## Development Files
- Intermediate CSVs are included for model selection, fine-tuning, and debugging.
- `adhoc.ipynb` and `labs.ipynb` are used for quick data manipulation (splitting, joining, etc.).
- **For future work:** Only new `.py` scripts are needed; notebooks are for prototyping.

## How to Use
1. Place your input data in `input_files/Diagnoses_list.csv`.
2. Run `embed_and_map_icd.py` to generate `output.csv`.
3. Split `output.csv` into chunks (see notebook or script for splitting).
4. Run `explanation.py` on each chunk to generate explanations in `flot/`.
5. Aggregate results as needed (see aggregation script or notebook).

## References
- See [ass.md](input_files/ass.md) for assignment details and ICD-10 resources.
- [all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Qwen2.5-0.5B-Instruct-medicalLLM-HuatuoGPT-o1-sft on HuggingFace](https://huggingface.co/bkholyday/Qwen2.5-0.5B-Instruct-medicalLLM-HuatuoGPT-o1-sft)


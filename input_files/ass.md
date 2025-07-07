# Assignemnet : ICD-10 Code Mapping AI System

### **Objective**:

Your task is to develop an AI-based system to map patient diagnoses to their corresponding ICD-10 codes. You will build AI system to automate the process. Document and justify your decisions clearly. You may use any tools , models or frameworks, such as Python, Hugging Face, or LangChain.

---

### **Background**:

**What is ICD-10?**

The **ICD-10 (International Classification of Diseases, 10th Revision)** is an international system used to classify and code diseases and health conditions. Each diagnosis is assigned a specific alphanumeric code, which ensures standardization in healthcare records across the world.

**Examples of ICD-10 Codes**:

- **E11.9**: **Diabetes mellitus type 2 without complications**
- **J45.909**: **Unspecified asthma, uncomplicated**
- **I10**: **Essential (primary) hypertension**
- **C34.90**: **Malignant neoplasm of unspecified part of the bronchus or lung**

---

### **Problem Statement**:

You are provided with a dataset of patient diagnoses. Each row represents list of  conditions diagnosed for a patient. Your task is to:

1. **Map Diagnoses to ICD-10 Codes and ICD_Description using available metadata (e.g., descriptions, codes) publicly as open source.**
2. **Handling Ambiguity**:For conditions that do not have a direct ICD-10 match, provide alternative suggestions and explain your choice.
3. **Justifications** For each diagnosis and ICD-10 code mapping, provide a brief explanation of why you selected the specific code.
4. **Deliverables**:
    - A final dataset including:
        - **Original diagnosis**
        - **Mapped ICD-10 code**
        - **Diagnosis description**

## Dataset

https://docs.google.com/spreadsheets/d/1O2wW-wQukh2F2o4_w7AmM2mS2wFwMGzYCdIGh54MT8Q/edit?gid=0#gid=0

## **Evaluation Criteria**:

- **Accuracy**: Correctness of the mappings to ICD-10 codes.
- **System Design**: Effectiveness and innovation in your AI system.

---

### **Resources**:

To assist you in solving the problem, here are a few useful tools and libraries:

1. **ICD-10 Code Database**:
    
    [ICD-10 Official Website](https://icd.who.int/browse10/2019/en) 
    
- **AAPC Guide to ICD-10**: American Academy of Professional Coders (AAPC) ICD-10 Guide – Provides an overview of ICD-10 coding, which might be helpful for candidates unfamiliar with the healthcare domain.
- https://github.com/icd-codex/icd-codex
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)
# Legal Document Analyzer – KB1 + KB2 Integration with JSON Output

## Overview
This project processes uploaded legal `.docx` files to:
- Identify the **legal process** involved
- Check for **required** and **missing documents**
- Detect potential **red flag issues**

It uses:
- **KB1** → Legal process classification & required documents
- **KB2** → Issue / red flag detection
- **FAISS** → Vector databases for similarity search
- **Ollama** → Local LLM for generating validated JSON output

The output is a **strict, clean JSON** object that can be consumed downstream or reviewed directly in the UI.

---

## 🔍 How It Works (Pipeline)

1. **Two Knowledge Bases**  
   - **KB1:** Legal processes & required documentation  
   - **KB2:** Common compliance issues & red flags  

2. **Vectorization with FAISS**  
   - Both KBs converted to FAISS vector databases using embeddings  

3. **Document Parsing & Embedding**  
   - `.docx` file parsed for text + embedding via `docx_parser_embedder.py`

4. **Combined Search**  
   - `JSONOutputGenerator` performs similarity search in **both KB1 and KB2**

5. **LLM Processing**  
   - Search results + document text fed into **Ollama** (local LLM)  
   - Prompt strictly enforces **valid JSON structure** (no nulls, no placeholders, no comments)

6. **Post‑processing & Validation**  
   - Removes placeholders `<...>`  
---

## Project Structure (Relevant Parts)
```
├── data/vector_dbs/
│ ├── kb1_faiss.index
│ ├── kb1_metadata.json
│ ├── kb2_faiss.index
│ ├── kb2_metadata.json
├── demo/
│ ├── json_output_screenshot.png
├── docx_parser_embedder.py
├── json_output.py
├── app.py # Gradio UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

Clone the repository
```
git clone <https://github.com/2CentsCapitalHR/ai-engineer-task-gurleenkaurbali19>
cd <repo_folder>
```
Install dependencies
```
pip install -r requirements.txt
```

**Additional Setup:**
- Install [Ollama](https://ollama.ai/) and ensure it runs locally  
- On Windows, Ollama CLI is expected at:  
  `C:\Users\<user>\AppData\Local\Programs\Ollama\ollama.exe`
- Place FAISS index and metadata files for KB1 & KB2 into `data/vector_dbs/`

---

##  How to Run
1. Upload a `.docx` file
2. Wait for FAISS + Ollama processing
3. View the **validated JSON output** in the UI
4. Output is saved to `final_output.json`

---

## 📌 Example Output

<img width="1918" height="962" alt="project_ss_json_file_output" src="https://github.com/user-attachments/assets/ef7d0826-8a54-47f4-8d22-d0098a733c2e" />


---




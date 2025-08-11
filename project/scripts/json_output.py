import os
import json
import faiss
import numpy as np
import subprocess
import re

class JSONOutputGenerator:
    def __init__(self,
                 kb1_index_path, kb1_metadata_path,
                 kb2_index_path, kb2_metadata_path,
                 model="phi3",
                 top_k=5,
                 ollama_path=None):

        # Absolute paths
        self.kb1_index_path = os.path.abspath(kb1_index_path)
        self.kb1_metadata_path = os.path.abspath(kb1_metadata_path)
        self.kb2_index_path = os.path.abspath(kb2_index_path)
        self.kb2_metadata_path = os.path.abspath(kb2_metadata_path)

        # Load FAISS indexes
        self.kb1_index = faiss.read_index(self.kb1_index_path)
        self.kb2_index = faiss.read_index(self.kb2_index_path)

        # Load metadata
        with open(self.kb1_metadata_path, "r", encoding="utf-8") as f:
            self.kb1_metadata = json.load(f)
        with open(self.kb2_metadata_path, "r", encoding="utf-8") as f:
            self.kb2_metadata = json.load(f)

        self.model = model
        self.top_k = top_k
        self.ollama_path = ollama_path or r"C:\Users\gurle\AppData\Local\Programs\Ollama\ollama.exe"

    def _search(self, index, metadata, query_embedding):
        """Run FAISS similarity search & retrieve top_k metadata entries."""
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array([query_embedding])
        else:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_embedding, self.top_k)
        results = []
        for idx in indices[0]:
            try:
                results.append(metadata[idx])
            except (IndexError, TypeError):
                results.append({})
        return results

    def _strip_json_comments(self, text: str) -> str:
        """Remove // and /* */ comments from JSON text."""
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text.strip()

    def build_prompt(self, parsed_doc_text, kb1_results, kb2_results):
        """Builds a combined prompt for Ollama with KB1 + KB2 context."""
        prompt = f"""
You are an AI legal agent that reviews corporate/legal documents for process completeness and red flag issues.

You will be given:
- KB1 similarity search results → legal process, classification, required documents.
- KB2 similarity search results → possible red flags & issues.
- Parsed text from the uploaded document.

STRICT RULES:
- Output ONLY one valid parsable JSON object — no explanations, no comments, no trailing commas.
- If a value is missing, use the string "Unknown" (not null or empty).
- Do NOT wrap values in < >.
- 'documents_uploaded' is always 1.
- Derive 'required_documents' from KB1 results if possible.
- Calculate 'missing_document' from required vs uploaded list; if unable, set to "Unknown".
- Use KB2 results to populate 'issues_found', at least one item — if none found, infer a likely generic issue.
- All issue fields must be non-empty strings.

JSON format:
{{
  "process": "<Detected Legal Process>",
  "documents_uploaded": 1,
  "required_documents": [ "<Doc1>", "<Doc2>" ],
  "missing_document": "<Name or 'Unknown'>",
  "issues_found": [
    {{
      "document": "<Doc Name>",
      "section": "<Section or 'Unknown'>",
      "issue": "<Description>",
      "severity": "<Low/Medium/High>",
      "suggestion": "<Suggested Fix>"
    }}
  ]
}}

KB1 similarity search results:
{json.dumps(kb1_results, indent=2, ensure_ascii=False)}

KB2 similarity search results:
{json.dumps(kb2_results, indent=2, ensure_ascii=False)}

Document snippet (first 2000 chars):
\"\"\"{parsed_doc_text[:2000]}\"\"\"
"""
        return prompt

    def call_ollama(self, prompt):
        """Call Ollama model via subprocess with UTF-8 decoding."""
        result = subprocess.run(
            [self.ollama_path, "run", self.model, prompt],
            capture_output=True
        )
        return result.stdout.decode("utf-8", errors="replace").strip()

    def _clean_and_validate(self, data: dict) -> dict:
        """Clean placeholder and null values from parsed JSON."""
        # Clean missing_document
        md = data.get("missing_document")
        if not md or "<" in str(md) or ">" in str(md):
            data["missing_document"] = str(md).replace("<", "").replace(">", "") if md else "Unknown"
            if not data["missing_document"].strip():
                data["missing_document"] = "Unknown"

        # Fix issues_found
        if "issues_found" in data:
            cleaned_issues = []
            for issue in data["issues_found"]:
                if not issue.get("document"):
                    issue["document"] = "Unknown"
                if not issue.get("section"):
                    issue["section"] = "Unknown"
                if not issue.get("issue"):
                    issue["issue"] = "Unknown"
                if not issue.get("severity"):
                    issue["severity"] = "Unknown"
                if not issue.get("suggestion"):
                    issue["suggestion"] = "Unknown"
                cleaned_issues.append(issue)
            if not cleaned_issues:
                cleaned_issues = [{
                    "document": "Unknown",
                    "section": "Unknown",
                    "issue": "No specific issue detected; review manually.",
                    "severity": "Medium",
                    "suggestion": "Manually review the document for compliance."
                }]
            data["issues_found"] = cleaned_issues

        return data

    def run(self, embedding_vector, parsed_text, output_json_path="final_output.json"):
        # Step 1: Search KB1 & KB2
        kb1_results = self._search(self.kb1_index, self.kb1_metadata, embedding_vector)
        kb2_results = self._search(self.kb2_index, self.kb2_metadata, embedding_vector)

        # Step 2: Build strict prompt
        prompt = self.build_prompt(parsed_text, kb1_results, kb2_results)

        # Step 3: Call Ollama
        llm_output = self.call_ollama(prompt)

        # Step 4: Extract JSON substring
        start = llm_output.find("{")
        end = llm_output.rfind("}") + 1
        json_str = llm_output[start:end] if start != -1 and end != -1 else llm_output

        # Step 5: Strip comments
        json_str = self._strip_json_comments(json_str)

        # Step 6: Parse & clean
        try:
            structured_data = json.loads(json_str)
            structured_data = self._clean_and_validate(structured_data)
        except json.JSONDecodeError:
            structured_data = {"raw_output": llm_output}

        # Step 7: Save output
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        return structured_data

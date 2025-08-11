import gradio as gr
from docx_parser_embedder import parse_and_embed
from json_output import JSONOutputGenerator
import json
import os

# --- Base directory for relative paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

# --- Initializing combined JSON output generator ---
json_generator = JSONOutputGenerator(
    kb1_index_path=os.path.join(BASE_DIR, "data", "vector_dbs", "kb1_faiss.index"),
    kb1_metadata_path=os.path.join(BASE_DIR, "data", "vector_dbs", "kb1_metadata.json"),
    kb2_index_path=os.path.join(BASE_DIR, "data", "vector_dbs", "kb2_faiss.index"),
    kb2_metadata_path=os.path.join(BASE_DIR, "data", "vector_dbs", "kb2_metadata.json"),
    model="phi3",
    ollama_path=r"C:\Users\gurle\AppData\Local\Programs\Ollama\ollama.exe"
)

def process_document(file_obj):
    """
    Parse the uploaded DOCX file, run similarity searches using KB1 and KB2,
    query Ollama for a structured JSON response, and return it.
    """
    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        parsed_text, embedding = parse_and_embed(file_path)

        # Run combined KB1+KB2 search and LLM call
        result_json = json_generator.run(
            embedding_vector=embedding,
            parsed_text=parsed_text,
            output_json_path=os.path.join(BASE_DIR, "final_output.json")
        )

        return json.dumps(result_json, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Legal Document Classifier, Missing Docs & Red Flag Checker (Combined KB1+KB2)")

    with gr.Row():
        file_input = gr.File(label="Upload a .docx file", file_types=[".docx"])

    with gr.Row():
        json_output = gr.Code(
            label="Processed Result (JSON Output)",
            language="json",
            interactive=False
        )

    file_input.change(fn=process_document, inputs=file_input, outputs=json_output)

if __name__ == "__main__":
    demo.launch()

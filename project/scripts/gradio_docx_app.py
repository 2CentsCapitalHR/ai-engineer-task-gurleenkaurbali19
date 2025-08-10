import tempfile
import gradio as gr
from docx_parser_embedder import parse_and_embed

def process_uploaded_file(file_obj):
    # Saving uploaded file to a temporary path
    tmp_path = file_obj.name if hasattr(file_obj, 'name') else file_obj


    try:
        # Parse + Embed
        text, embedding = parse_and_embed(tmp_path)

        # Prepare embedding preview for UI
        emb_preview = embedding[:10]  # Show first 10 values
        emb_shape = embedding.shape

        return (text[:1500] + "...",  # Show text preview in UI
                f"Shape: {emb_shape}, First 10 dims: {emb_preview}")
    except Exception as e:
        return f"Error: {str(e)}", ""

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("##Document Parser & Embedder Demo")
    with gr.Row():
        file_input = gr.File(label="Upload a .docx file", file_types=[".docx"])
    with gr.Row():
        text_output = gr.Textbox(label="Extracted Text (Preview)", lines=20)
    with gr.Row():
        emb_output = gr.Textbox(label="Embedding Vector (Preview)")

    file_input.change(fn=process_uploaded_file,
                      inputs=file_input,
                      outputs=[text_output, emb_output])

if __name__ == "__main__":
    demo.launch()

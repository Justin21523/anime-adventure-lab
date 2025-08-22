import gradio as gr
def echo(x): return x
with gr.Blocks() as demo:
    gr.Markdown("# SagaForge WebUI")
    inp = gr.Textbox(label="Prompt"); out = gr.Textbox(label="Reply")
    inp.submit(echo, inp, out)
demo.queue().launch(server_name="0.0.0.0", server_port=7860)

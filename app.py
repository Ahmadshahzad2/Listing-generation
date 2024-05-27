import gradio as gr
import tempfile
from PIL import Image
from pipeline import main

# Function to format the output
def format_output(data):
    formatted_data = []
    for item in data:
        block = f"**{item['Title']}**\n\n" + "\n".join([f"- {feature}" for feature in item['Features']])
        formatted_data.append(block)
    return formatted_data

# Function to handle image input, save it temporarily, and display formatted output
def process_image(image):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    # Process the image using your main function
    data = main(temp_file_path)
    formatted_data = format_output(data)
    return tuple(formatted_data)  # Returning as a tuple for Gradio's multiple outputs

# Create Gradio blocks for each dictionary
with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image",  image_mode="RGB", height=512, width=512)

    with gr.Row():
        output1 = gr.Markdown(label="Block 1")
        output2 = gr.Markdown(label="Block 2")
        output3 = gr.Markdown(label="Block 3")

    # Button to trigger the display function
    button = gr.Button("Process Image")
    button.click(process_image, inputs=input_image, outputs=[output1, output2, output3])

demo.launch()

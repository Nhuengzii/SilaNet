import gradio as gr
import torch
from models import GeneratorNetwork
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

# Load your model
device = torch.device("cpu")
generator = GeneratorNetwork()
generator.load_state_dict(torch.load("./checkpoint/generator_sila_1000.pt", map_location=device))
generator = generator.to(device)

# Define the function to generate images with random noise

def generate_image(s):
    fixed_noise = torch.randn((4, 100, 1, 1)).to(device)
    noise = fixed_noise + s
    with torch.no_grad():
        images = generator(noise)
        grid = torchvision.utils.make_grid(images, normalize=True, nrow=2)
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig("./gen.jpg")
        plt.close(fig)
    return np.transpose(grid.numpy(), (1, 2, 0))


# Create the interface
iface = gr.Interface(
        generate_image,
        gr.inputs.Slider(-0.5, 0.5, 0.01, label="Seed"),
        "image",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)

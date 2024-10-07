from io import BytesIO
import requests
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import os
from torchvision import transforms
import wandb  # Import wandb
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize WandB
wandb.init(project="dog-image-generation", name="standing_to_sitting")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion pipeline
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
text2img_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

img2img_pipe.to(device)
text2img_pipe.to(device)

# Function: Convert PIL image to tensor
def image_to_tensor(image_path):
    init_image = Image.open(BytesIO(image_path)).convert("RGB")
    init_image = init_image.resize((768, 512))
    transform = transforms.ToTensor()  # Default converts values to [0, 1]
    init_image_tensor = transform(init_image)
    return init_image_tensor

# Function: Generate latent vectors
def img2img_generate_latent(prompt, init_image_tensor, num_images=1):
    init_image_latent = img2img_pipe.image_processor.preprocess(init_image_tensor).to(device)
    
    with torch.no_grad():
        for _ in range(num_images):
            output = img2img_pipe(prompt, image=init_image_latent, num_inference_steps=3, guidance_scale=7.5, output_type="latent")
            latent_vector = output["images"]  # Assume we can get latent vectors here
            latent_vector = latent_vector.detach().to(device)
    
    return latent_vector

def text_generate_latents(prompt, num_images=1):
    latents_list = []
    
    with torch.no_grad():
        for _ in range(num_images):
            output = text2img_pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="latent")
            latent_vector = output["images"]  # Assume we can get latent vectors here
            latent_vector = latent_vector.detach().to(device)
            latents_list.append(latent_vector)
    
    latents_stack = torch.stack(latents_list)
    return latents_stack

# Read initial image and convert to tensor
img_url = "https://elaine3240.wordpress.com/wp-content/uploads/2019/06/e69fb4e78aac-e9bb91e889b2.jpg"
response = requests.get(img_url)
init_image_tensor = image_to_tensor(response.content).to(device)

# Generate latent vectors for two categories
latents_standing = text_generate_latents("A photo of a standing dog", num_images=150)
latents_sitting = text_generate_latents("A photo of a sitting dog", num_images=150)

# Calculate mean (centroid)
mean_standing = torch.mean(latents_standing, dim=0)
mean_sitting = torch.mean(latents_sitting, dim=0)

# Calculate direction vector (from standing to sitting)
direction_vector = mean_sitting - mean_standing

# Calculate the variability within each class (standard deviation)
std_standing = torch.std(latents_standing, dim=0)
std_sitting = torch.std(latents_sitting, dim=0)

# Calculate inter-class variability (mean difference)
inter_class_variability = torch.abs(mean_sitting - mean_standing)

# Set thresholds
class_within_threshold_standing = std_standing < 0.1
class_within_threshold_sitting = std_sitting < 0.1

# Create mask to retain high inter-class variability and zero out low intra-class variability
final_mask = (inter_class_variability > 0.1) & ~(class_within_threshold_standing | class_within_threshold_sitting)

# Filter direction vector
direction_vector_filtered = direction_vector.clone()
direction_vector_filtered[~final_mask] = 0

# Print information about the direction vector
direction_label = "From standing to sitting"
print(f"Direction vector {direction_label} (after filtering):")
print(direction_vector_filtered)

# Add the direction vector to the initial image's latent vector
init_image_latent = img2img_pipe.image_processor.preprocess(init_image_tensor).to(device)
init_latents = retrieve_latents(img2img_pipe.vae.encode(init_image_latent), generator=None)

import torch.nn.functional as F
tensor_a_expanded = F.interpolate(direction_vector_filtered, size=(64, 96), mode='bilinear', align_corners=False)
adjusted_latent_vector = init_latents + 20 * tensor_a_expanded

init_image_latent = img2img_pipe.image_processor.preprocess(init_image_tensor).to(device)
# Generate image with adjusted latent vector
output = img2img_pipe('A photo of a sitting dog', image=init_image_latent, num_inference_steps=150, guidance_scale=1.5, output_type="latent")

# 获取生成图像的潜在向量
latent_vector_output = output["images"]  # Assume we can get latent vectors here
latent_vector_output = latent_vector_output.detach()  # 从计算图中分离

# 确保 latent_vector 在同一设备上
latent_vector_output = latent_vector_output.to(device)

# Decode the adjusted latent vector and generate a new image
with torch.no_grad():
    generated_image = img2img_pipe.vae.decode(latent_vector_output / img2img_pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * generated_image.shape[0]
    final_image = img2img_pipe.image_processor.postprocess(generated_image, output_type="pil", do_denormalize=do_denormalize)[0]

# Save the generated image
final_image.save("img2img_adjusted_A_photo_of_a_standing_dog.png")

# 将 final_image 转换为 NumPy 数组
final_image_np = np.array(final_image)

# 确保形状为 (H, W, C)
if final_image_np.ndim == 3 and final_image_np.shape[0] == 3:  # 如果是 (C, H, W)
    final_image_np = final_image_np.transpose(1, 2, 0)  # 转换为 (H, W, C)

# 现在将其传递给 wandb
wandb.log({"final_image": wandb.Image(final_image_np)})

init_image_numpy = init_image_tensor.cpu().numpy()

if init_image_numpy.ndim == 3 and init_image_numpy.shape[0] == 3:  # 如果是 (C, H, W)
    init_image_numpy = init_image_numpy.transpose(1, 2, 0)  # 转换为 (H, W, C)

# Upload the initial image
wandb.log({"initial_image": wandb.Image(init_image_numpy)})

# Visualize the final generated image
final_image.show()

# End WandB logging
wandb.finish()


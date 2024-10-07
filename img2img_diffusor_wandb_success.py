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
from sklearn.manifold import TSNE
# Initialize WandB
wandb.init(project="dog-image-generation", name="local_standing_to_sitting_data_analysize")


class Config:
    def __init__(self):
        self.sampleTimes = 5
        self.target_prompt = "A photo of a sitting dog"
        self.source_prompt = "A photo of a standing dog"
        self.model_name = "standing_to_sitting"
        self.pretrained_model = "sd-legacy/stable-diffusion-v1-5"
        self.direction_weight = 0
        self.i2i_guidance_scale = 7.5
        self.image_path = "https://elaine3240.wordpress.com/wp-content/uploads/2019/06/e69fb4e78aac-e9bb91e889b2.jpg"
        self.t2i_inference_steps = 50
        self.i2i_inference_steps = 350


config = Config()
wandb.config.update(config.__dict__)  # 将对象的字典更新到 wandb.config


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion pipeline
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(config.pretrained_model)
text2img_pipe = StableDiffusionPipeline.from_pretrained(config.pretrained_model)

img2img_pipe.to(device)
text2img_pipe.to(device)

# 使用 t-SNE 降维
def reduce_latents_to_2D(latents):
    tsne = TSNE(n_components=2,perplexity=1, random_state=42)
    reduced_data = tsne.fit_transform(latents.cpu().numpy())
    return reduced_data

# Function: Convert PIL image to tensor
def image_to_tensor(image_path):
    init_image = Image.open(BytesIO(image_path)).convert("RGB")
    init_image = init_image.resize((768, 512))
    transform = transforms.ToTensor()  # Default converts values to [0, 1]
    init_image_tensor = transform(init_image)
    return init_image_tensor

# Function: Generate latent vectors
def img2img_generate_latent(prompt, init_image_tensor, num_images=5):
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
            output = text2img_pipe(prompt, num_inference_steps=config.t2i_inference_steps, guidance_scale=7.5, output_type="latent")
            latent_vector = output["images"]  # Assume we can get latent vectors here
            latent_vector = latent_vector.detach().to(device)
            latents_list.append(latent_vector)
    
    latents_stack = torch.stack(latents_list)
    return latents_stack

# Read initial image and convert to tensor
# img_url = "https://elaine3240.wordpress.com/wp-content/uploads/2019/06/e69fb4e78aac-e9bb91e889b2.jpg"
response = requests.get(config.image_path)
init_image_tensor = image_to_tensor(response.content).to(device)

# Generate latent vectors for two categories
latents_standing = text_generate_latents(config.source_prompt, num_images=config.sampleTimes)
latents_sitting = text_generate_latents(config.target_prompt, num_images=config.sampleTimes)


latents_standing_flat = latents_standing.view(latents_standing.size(0), -1)  # 形状变为 (num_samples, channels * height * width)
latents_sitting_flat = latents_sitting.view(latents_sitting.size(0), -1)      # 同上
print('latents_standing_flat_size',latents_standing_flat.size())
print('latents_sitting_flat_size',latents_sitting_flat.size())

# 降维
reduced_standing = reduce_latents_to_2D(latents_standing_flat)
reduced_sitting = reduce_latents_to_2D(latents_sitting_flat)

# # 合并数据并生成标签
# combined_data = np.vstack((reduced_standing, reduced_sitting))
# labels = ["Standing"] * reduced_standing.shape[0] + ["Sitting"] * reduced_sitting.shape[0]

# # 创建数据表
# table = wandb.Table(data=combined_data.tolist(), columns=["t2I Standing", "t2I Sitting"])
# table.add_column("Label", labels)

# # 记录降维后的数据到 W&B
# wandb.log({
#     "2D t-SNE Scatter Plot": wandb.plot.scatter(
#         table,
#         "t2I Standing",
#         "t2I Sitting",
#         title="T2I Standing vs Sitting Latents"
#     )
# })
# Calculate mean (centroid)
mean_standing = torch.mean(latents_standing, dim=0)
mean_sitting = torch.mean(latents_sitting, dim=0)


reduce_mean_standing = mean_standing.view(1, -1)
reduce_mean_sitting = mean_sitting.view(1, -1)

import matplotlib.pyplot as plt
# 记录降维后的数据到 W&B
plt.figure(figsize=(8, 6))
plt.scatter(reduced_standing[:, 0], reduced_standing[:, 1], label='Standing', alpha=0.5)
plt.scatter(reduced_sitting[:, 0], reduced_sitting[:, 1], label='Sitting', alpha=0.5)
plt.scatter(reduce_mean_standing[0,0].cpu().numpy(), reduce_mean_standing[0,1].cpu().numpy(), c='red', marker='X', s=200, label='Mean Standing')
plt.scatter(reduce_mean_sitting[0,0].cpu().numpy(), reduce_mean_sitting[0,1].cpu().numpy(), c='blue', marker='X', s=200, label='Mean Sitting')
plt.title("origin_standing_to_sitting t-SNE 2D Scatter Plot with Means")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid()

# 保存图像并上传到 W&B
plt.savefig("origin_sitting_and_standing_tsne_plot.png")
wandb.log({"origin_sitting_and_standing t-SNE Plot": wandb.Image("origin_sitting_and_standing_tsne_plot.png")})

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

# 应用过滤
filtered_latents_standing = latents_standing[:, final_mask]
filtered_latents_sitting = latents_sitting[:, final_mask]

filtered_latents_standing_flat = latents_standing.view(filtered_latents_standing.size(0), -1)  # 形状变为 (num_samples, channels * height * width)
filtered_latents_sitting_flat = latents_sitting.view(filtered_latents_sitting.size(0), -1)  

# 降维
filtered_reduced_standing = reduce_latents_to_2D(filtered_latents_standing_flat)
filtered_reduced_sitting = reduce_latents_to_2D(filtered_latents_sitting_flat)


plt.figure(figsize=(8, 6))
plt.scatter(filtered_reduced_standing[:, 0], filtered_reduced_standing[:, 1], label='Standing', alpha=0.5)
plt.scatter(filtered_reduced_sitting[:, 0], filtered_reduced_sitting[:, 1], label='Sitting', alpha=0.5)
# plt.scatter(reduce_mean_standing[0,0].cpu().numpy(), reduce_mean_standing[0,1].cpu().numpy(), c='red', marker='X', s=200, label='Mean Standing')
# plt.scatter(reduce_mean_sitting[0,0].cpu().numpy(), reduce_mean_sitting[0,1].cpu().numpy(), c='blue', marker='X', s=200, label='Mean Sitting')
plt.title("filter_standing_to_sitting t-SNE 2D Scatter Plot with Means")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid()

# 保存图像并上传到 W&B
plt.savefig("filter_sitting_and_standing_tsne_plot.png")
wandb.log({"filter_sitting_and_standing t-SNE Plot": wandb.Image("filter_sitting_and_standing_tsne_plot.png")})

# 合并数据并生成标签
# filtered_combined_data = np.vstack((filtered_reduced_standing, filtered_reduced_sitting))
# filtered_labels = ["Standing"] * filtered_reduced_standing.shape[0] + ["Sitting"] * filtered_reduced_sitting.shape[0]

# # 创建数据表
# filter_table = wandb.Table(data=filtered_combined_data.tolist(), columns=["filter t2I Standing", "filter t2I Sitting"])
# filter_table.add_column("filtered_labels", filtered_labels)

# # 记录降维后的数据到 W&B
# wandb.log({
#     "filter standing-> sitting 2D t-SNE Scatter Plot": wandb.plot.scatter(
#         filter_table,
#         "t2I Standing",
#         "t2I Sitting",
#         title="Filter T2I Standing vs Sitting Latents"
#     )
# })
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
init_latents = img2img_pipe.vae.config.scaling_factor * init_latents


import torch.nn.functional as F
tensor_a_expanded = F.interpolate(direction_vector_filtered, size=(64, 96), mode='bilinear', align_corners=False)
adjusted_latent_vector = init_latents + config.direction_weight

# Generate image with adjusted latent vector
output = img2img_pipe(config.target_prompt, image=adjusted_latent_vector, num_inference_steps=config.i2i_inference_steps, guidance_scale=config.i2i_guidance_scale, output_type="latent")

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


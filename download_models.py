## Download Available Model
import os
import subprocess

root_dir = "~/sd-train"
pretrained_model = os.path.join(root_dir, "pretrained_model")
os.chdir(root_dir)

models = {
    "AnyLoRA_noVae_fp16-pruned": "https://huggingface.co/Lykon/AnyLoRA/resolve/main/AnyLoRA_noVae_fp16-pruned.safetensors",
    "AAM_Anylora_AnimeMix": "https://huggingface.co/Lykon/AnyLoRA/resolve/main/AAM_Anylora_AnimeMix.safetensors",
}

v2_models = {
    "v2-1_512-ema-pruned": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors",
    "v2-1_768-ema-pruned": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors",
}

installModels = []
installv2Models = []

### SD1.x model
model_name = "AnyLoRA_noVae_fp16-pruned"  # ["", "AnyLoRA_noVae_fp16-pruned", "AAM_Anylora_AnimeMix"]
### SD2.x model
v2_model_name = ""  # ["", "v2-1_512-ema-pruned", "v2-1_768-ema-pruned"]
# Change this part with your own huggingface token if you need to download your private model
hf_token = ""

if model_name:
    model_url = models.get(model_name)
    if model_url:
        installModels.append((model_name, model_url))

if v2_model_name:
    v2_model_url = v2_models.get(v2_model_name)
    if v2_model_url:
        installv2Models.append((v2_model_name, v2_model_url))


def install(checkpoint_name, url):
    ext = "ckpt" if url.endswith(".ckpt") else "safetensors"

    user_header = f'"Authorization: Bearer {hf_token}"'
    subprocess.run(f'aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {pretrained_model} -o {checkpoint_name}.{ext} "{url}"', shell=True)


def install_checkpoint():
    for model in installModels:
        install(model[0], model[1])
    for v2model in installv2Models:
        install(v2model[0], v2model[1])


install_checkpoint()

input("Press the Enter key to continue: ")


## Download Custom Model
os.chdir(root_dir)

### Custom model
modelUrls = ""  # Comma-separated

def install(url):
    base_name = os.path.basename(url)

    if "drive.google.com" in url:
        os.chdir(pretrained_model)
        subprocess.run(f"gdown --fuzzy {url}")
    elif "huggingface.co" in url:
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        user_header = f'"Authorization: Bearer {hf_token}"'
        subprocess.run(f'aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {pretrained_model} -o {base_name} "{url}"')
    else:
        subprocess.run(f'aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {pretrained_model} "{url}"')

if modelUrls:
    urls = modelUrls.split(",")
    for url in urls:
        install(url.strip())

input("Press the Enter key to continue: ")


## Download Available VAE (Optional)
os.chdir(root_dir)
vae_dir = os.path.join(root_dir, "vae")

vaes = {
    "none": "",
    "any.vae.safetensors": "https://huggingface.co/NoCrypt/resources/resolve/main/VAE/any.vae.safetensors",
    "wd.vae.safetensors": "https://huggingface.co/NoCrypt/resources/resolve/main/VAE/wd.vae.safetensors",
    "vae-ft-mse-840000-ema-pruned.safetensors": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
    "blessed2.vae.safetensors": "https://huggingface.co/NoCrypt/resources/resolve/main/VAE/blessed2.vae.safetensors"
}
install_vaes = []

# Select one of the VAEs to download, select `none` for not download VAE:
vae_name = "none"  # ["none", "any.vae.safetensors", "wd.vae.safetensors", "vae-ft-mse-840000-ema-pruned.safetensors", "blessed2.vae.safetensors"]

if vae_name in vaes:
    vae_url = vaes[vae_name]
    if vae_url:
        install_vaes.append((vae_name, vae_url))


def install(vae_name, url):
    user_header = f'"Authorization: Bearer {hf_token}"'
    subprocess.run(f'aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {vae_dir} -o {vae_name} "{url}"')


def install_vae():
    for vae in install_vaes:
        install(vae[0], vae[1])


install_vae()

import torch
from vae import VAE 
from torchvision.utils import save_image


vae = VAE(in_channels=3, latent_dim=8, hidden_dims=None)

device = torch.device('cpu')
vae = torch.load("encoder_decoder/logs/mnist_vae.pt", map_location=torch.device('cpu')) 
print(vae) 

num_samples = 100 
samples = vae.sample(num_samples=num_samples, current_device=device) 
for i, s in enumerate(samples): 
    print(i)
    save_image(s, 'encoder_decoder/logs/samples/img_'+str(i)+'.png')

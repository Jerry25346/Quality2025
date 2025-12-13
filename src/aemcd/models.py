import torch
import torch.nn as nn

# Gussian noise layer
class GaussianNoise(nn.Module):
    """Gaussian noise layer for data augmentation and overfitting prevention"""
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std 
            return x + noise
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim =6):
        super(Encoder, self).__init__()
        # define layers 
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, latent_dim) # Final layer without activation
        )
        

    def forward(self, x):
        return self.network(x)

#Decoder    
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            

            nn.Linear(32, 64),
            nn.ReLU(),
            

            nn.Linear(64, output_dim) # Final layer without activation
        )
    
    def forward(self, z):
        return self.network(z)  
    
# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=6, noise_std=0.03):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.noise = GaussianNoise(noise_std)

    def forward(self, x):
        noisy_x = self.noise(x)   # When training add noise
        z = self.encoder(noisy_x)
        reconstructed_x = self.decoder(z) 
        return reconstructed_x ,z
    
    def encode(self, x):
        self.eval()    ## dropout off
        with torch.no_grad(): # no gardiant
            return self.encoder(x)

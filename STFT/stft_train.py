import torch
import torch.optim as optim
# Assuming STFT class is in a file named my_stft_module.py and util.py in same directory
from stft import STFTLEARN
import torch.functional as F

# Example Usage
# Generate some dummy audio data
batch_size = 4
sample_rate = 16000
duration = 1  # seconds
num_samples = sample_rate * duration
dummy_audio = torch.randn(batch_size, num_samples)

# Initialize STFT with learnable basis and window
# You can set learn_basis and/or learn_window to True
stft_model = STFTLEARN(filter_length=1024, hop_length=512, learn_basis=True, learn_window=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stft_model.to(device)
dummy_audio = dummy_audio.to(device)

# Define an optimizer
# The optimizer will automatically find all torch.nn.Parameter objects
# within stft_model and update them during training.
optimizer = optim.Adam(stft_model.parameters(), lr=0.001)

# Example: A simple training loop (e.g., for autoencoder type reconstruction)
num_epochs = 1
for epoch in range(num_epochs):
    optimizer.zero_grad() # Clear gradients

    # Forward pass: audio -> STFT -> inverse STFT (reconstruction)
    reconstructed_audio = stft_model(dummy_audio)

    # Define a loss function (e.g., Mean Squared Error for reconstruction)
    loss = F.mse_loss(reconstructed_audio, dummy_audio)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# After training, you can access the learned parameters
print("\nLearned Forward Basis (first filter):")
print(stft_model.forward_basis[0, 0, :].data) # Access a slice to see part of it

print("\nLearned Window:")
print(stft_model.fft_window.data)
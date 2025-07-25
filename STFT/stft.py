import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from .util import window_sumsquare


def window_sumsquare_torch(window, n_frames, hop_length, win_length, n_fft):
    """
    Calculate the sum-square envelope of the window function over time,
    to be used for reconstruction normalization.
    """
    window_square = window.pow(2)
    window_square_padded = F.pad(window_square, (0, n_fft - win_length))  # shape [n_fft]
    total_len = n_fft + hop_length * (n_frames - 1)
    sum_square = window.new_zeros(total_len)

    for i in range(n_frames):
        start = i * hop_length
        sum_square[start:start + n_fft] += window_square_padded

    return sum_square

# our class that is learnable
class STFTLEARN(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann', learn_basis=False, learn_window=False,learn_fft=False):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
            learn_basis {bool} -- If True, the Fourier basis will be learned. (default: {False})
            learn_window {bool} -- If True, the window function will be learned. (default: {False})
        """
        super(STFTLEARN, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window_type = window # Renamed to avoid conflict with potential learned window parameter
        #self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        self.learn_window=learn_window
        # 1. Initialize Fourier Basis
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        # Convert to Torch Tensor
        forward_basis_init = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis_init = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        # Make basis learnable if specified
        if learn_fft:
            print('cache6 ok')
            self.register_buffer('initial_forward_basis', forward_basis_init.clone())
            #self.forward_basis = torch.nn.Parameter(forward_basis_init,requires_grad=True)
            self.forward_basis = torch.nn.Parameter(forward_basis_init.clone(),requires_grad=True)
            self.inverse_basis = torch.nn.Parameter(inverse_basis_init.clone(),requires_grad=True)
            
        
        else:
            self.register_buffer('forward_basis', forward_basis_init.float())
            self.register_buffer('inverse_basis', inverse_basis_init.float())

    
        assert(filter_length >= self.win_length)

        # 2. Initialize Window Function
        fft_window_init = get_window(self.window_type, self.win_length, fftbins=True)
        # print('Warning removed argument in pad_center') # This warning can be removed if not relevant for your use case
        fft_window_init = pad_center(data=fft_window_init, size=filter_length, mode='constant')
        
        fft_window_init_tensor = torch.from_numpy(fft_window_init).float()

        if learn_window:
            
            # Store the initial state as a buffer to ensure it moves with the model
            self.register_buffer('initial_fft_window', fft_window_init_tensor.clone())
            self.fft_window = torch.nn.Parameter(fft_window_init_tensor.clone(), requires_grad=True)
        else:
            # If not learning, it remains a buffer
            self.register_buffer('fft_window', fft_window_init_tensor.clone())
            self.initial_fft_window = None # Not applicable if not learnable
    

    def enforce_forward_basis_constraints(self, delta=0.001):
        if hasattr(self, 'initial_forward_basis') and self.initial_forward_basis is not None:
         
            lower_bound = self.initial_forward_basis - delta
            upper_bound = self.initial_forward_basis + delta
            with torch.no_grad():
                self.forward_basis.copy_(torch.clamp(self.forward_basis, min=lower_bound, max=upper_bound))

    def enforce_fft_window_constraints(self, delta=0.001):
        if self.learn_window and self.initial_fft_window is not None:
            lower_bound = self.initial_fft_window - delta
            upper_bound = self.initial_fft_window + delta
            with torch.no_grad():
                self.fft_window.copy_(torch.clamp(self.fft_window, min=lower_bound, max=upper_bound))


    def get_fft_basis(self):
        return self.forward_basis
    

    def get_window_init(self):
        # will be relevant only if learn_window is true
        return self.fft_window
    
    def forward(self, input_data):
        return self.transform(input_data=input_data)


    def transform(self, input_data):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        # Apply window to forward_basis before convolution (if window is trainable, this is crucial)
        # Ensure that the dimensions for multiplication are correct (broadcasting rules)
        # self.forward_basis has shape [num_filters, 1, filter_length]
        # self.fft_window has shape [filter_length]
        # We need to reshape self.fft_window to [1, 1, filter_length] for broadcasting
        windowed_forward_basis = self.forward_basis * self.fft_window.view(1, 1, -1)


        forward_transform = F.conv1d(
            input_data,
            windowed_forward_basis, # Use the windowed basis here
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        # Detach phase from the computation graph if you don't need to backpropagate through it.
        # However, for full differentiability, it's often kept. For simplicity, we'll keep it for now.
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase
    
    

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        # Apply window to inverse_basis before transpose convolution
        windowed_inverse_basis = self.inverse_basis * self.fft_window.view(1, 1, -1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            windowed_inverse_basis, # Use the windowed basis here
            stride=self.hop_length,
            padding=0)

       
        if self.learn_window : 
           
            window_sum = window_sumsquare_torch(
                self.fft_window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)

            # Only divide by non-zero sums to avoid NaNs/Infs
            if approx_nonzero_indices.numel() > 0:
                inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform

    # def forward(self, input_data):
    #     """Take input data (audio) to STFT domain and then back to audio.

    #     Arguments:
    #         input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

    #     Returns:
    #         reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
    #             shape (num_batch, num_samples)
    #     """
    #     magnitude, phase = self.transform(input_data) # Removed self. assignment as it's not strictly necessary for forward pass
    #     reconstruction = self.inverse(magnitude, phase)
    #     return reconstruction
    
    

# initial code

class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann'):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).
        
        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris) 
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        #print('Warning removed argument in pad_center')
        fft_window = pad_center(data=fft_window,size=filter_length,mode='constant')
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """Take input data (audio) to STFT domain.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        
        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced 
        by the ```transform``` function.
        
        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        
        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
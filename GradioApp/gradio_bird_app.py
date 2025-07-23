
import os
import numpy as np
from collections import Counter
import librosa
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from scipy.signal import get_window
from librosa.util import pad_center
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import parametrize
from librosa.filters import mel as librosa_mel_fn
import random
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


dataset_root_folder='/home/christophe/birdclef/'
# contains audio files from birds
audio_input_folder = dataset_root_folder+"train_audio"


folder_store_audio=dataset_root_folder+audio_input_folder
label_list = sorted(os.listdir(os.path.join(dataset_root_folder, 'train_audio')))
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

def get_id_from_label(dict_label:dict,label:str):
    return dict_label[label]

id_test=get_id_from_label(label2id,'asikoe2')
assert id_test==4
print('asikoe2 index: ',id_test)
id_test=get_id_from_label(label2id,'barswa')
assert id_test==9
print('barswa index: ',id_test)


FS=32000
FIXED_LENGTH = int(5.0 * FS)  # 5 seconds

def get_audio_data(audio_raw_file):
    audio_data, _ = librosa.load(audio_raw_file, sr=FS)
    if len(audio_data) < FIXED_LENGTH:
        
        # Pad with zeros
        pad_width = FIXED_LENGTH - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    else:
        
        # Truncate
        audio_data = audio_data[:FIXED_LENGTH]

    return torch.from_numpy(audio_data).unsqueeze(0)
 
# V2 dataset much faster than v1
# augemted spectrograms are computed and stored on the disk in a specific folder
class BirdAudioDatasetV2(Dataset):
    def __init__(self, root_dir, audio_extensions=(".ogg",)):
        self.root_dir = root_dir
        self.audio_extensions = audio_extensions
        self.samples = []
        self.label_counts = Counter()

        # Walk through each subdirectory
        for label in os.listdir(root_dir):
            current_label_samples = 0 
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for fname in os.listdir(label_path):
                if fname.lower().endswith(audio_extensions):

                    fpath = os.path.join(label_path, fname)
                    self.samples.append((fpath, label))
                    current_label_samples += 1
            self.label_counts[label] = current_label_samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        input_model=get_audio_data(audio_path)

        id_label=get_id_from_label(dict_label=label2id,label=label)
        return input_model, id_label
    
    def count_samples_per_label(self):
       
        return dict(self.label_counts)
    

# it's important to bound varaiations of FFT during training
# too big varaitions will lead to poor results
class BoundFourierBasis(nn.Module):
    def __init__(self, base_param, delta=0.01):
        super().__init__()
        self.register_buffer('base_param', base_param)
        self.delta = delta

    def forward(self, X):
        return torch.clamp(X, self.base_param - self.delta, self.base_param + self.delta)

# same approach for windowing 
class BoundWindowParam(nn.Module):
    def __init__(self, base_param, delta=0.01):
        super().__init__()
        self.register_buffer('base_param', base_param)
        self.delta = delta

    def forward(self, X):
        return torch.clamp(X, self.base_param - self.delta, self.base_param + self.delta)


# our class that is learnable
class STFTLEARN(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann', learn_window=False,learn_fft=False):
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
        self.window_type = window 
        
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        self.learn_window=learn_window
        self.learn_fft=learn_fft
       
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        # Convert to Torch Tensor
        self.forward_basis_init = torch.FloatTensor(fourier_basis[:, None, :])
        
        self.inverse_basis_init = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        # Make basis learnable if specified
        if self.learn_fft:

            self.register_buffer('initial_forward_basis', self.forward_basis_init.clone())
            self.forward_basis = torch.nn.Parameter(self.initial_forward_basis.clone(), requires_grad=True)
            parametrize.register_parametrization(
                    self, "forward_basis", BoundFourierBasis(self.forward_basis_init, delta=0.01)
                    )
            # Parametrize the forward basis
            # parametrize.register_parametrization(self, "forward_basis",
            #                              BoundFourierBasis(self.forward_basis, self.forward_basis_init, delta=0.01))

            # self.register_buffer('initial_forward_basis', self.forward_basis_init.clone())
            # self.forward_basis = torch.nn.Parameter(self.initial_forward_basis.clone(),requires_grad=True)
            # self.inverse_basis = torch.nn.Parameter(self.initial_forward_basis.clone(),requires_grad=True)
            # # self.forward_basis = torch.nn.Parameter(self.forward_basis_init.clone(),requires_grad=True)
            # # self.inverse_basis = torch.nn.Parameter(self.inverse_basis_init.clone(),requires_grad=True)

            # parametrize.register_parametrization(self, "forward_basis", BoundFourierBasis(self.forward_basis,self.forward_basis_init))
            
             
        else:

            self.register_buffer('initial_forward_basis', self.forward_basis_init.clone())
            self.forward_basis = torch.nn.Parameter(self.initial_forward_basis.clone(), requires_grad=False)
            parametrize.register_parametrization(
                    self, "forward_basis", BoundFourierBasis(self.forward_basis_init, delta=0.0)
                    )
            # self.register_buffer('initial_forward_basis', self.forward_basis_init.clone())
            # self.forward_basis = torch.nn.Parameter(self.initial_forward_basis.clone(), requires_grad=True)
    
            # # Parametrize the forward basis
            # parametrize.register_parametrization(self, "forward_basis",
            #                              BoundFourierBasis(self.forward_basis, self.forward_basis_init, delta=0.0))
            # self.register_buffer('forward_basis', self.forward_basis_init.float())
            # self.register_buffer('inverse_basis',self.inverse_basis_init.float())

    
        assert(filter_length >= self.win_length)

        # 2. Initialize Window Function
        fft_window_init = get_window(self.window_type, self.win_length, fftbins=True)
        
        fft_window_init = pad_center(data=fft_window_init, size=filter_length, mode='constant')
        
        self.fft_window_init_tensor = torch.from_numpy(fft_window_init).float()

        if self.learn_window:

             self.register_buffer('initial_fft_window', self.fft_window_init_tensor.clone())
             self.fft_window = torch.nn.Parameter(self.initial_fft_window.clone(), requires_grad=True)

            # Add constraint: clip deviations from the init window
             parametrize.register_parametrization(
                self, "fft_window", BoundWindowParam(self.fft_window_init_tensor, delta=0.01)
                    )

            
            # self.register_buffer('initial_fft_window', self.fft_window_init_tensor.clone())
            # #self.fft_window = torch.nn.Parameter(self.fft_window_init_tensor.clone(), requires_grad=True)
            # self.fft_window = torch.nn.Parameter(self.initial_fft_window.clone(), requires_grad=True)

        else:

            self.register_buffer('initial_fft_window', self.fft_window_init_tensor.clone())
            self.fft_window = torch.nn.Parameter(self.initial_fft_window.clone(), requires_grad=False)

            # Add constraint: clip deviations from the init window
            parametrize.register_parametrization(
                self, "fft_window", BoundWindowParam(self.fft_window_init_tensor, delta=0.0)
                    )



            # If not learning, it remains a buffer
            # self.register_buffer('fft_window', self.fft_window_init_tensor.clone())
            # self.initial_fft_window = None # Not applicable if not learnable
    

    def enforce_forward_basis_constraints(self, delta=0.001):
        if self.learn_fft and self.initial_forward_basis is not None:
         
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

        # We need to reshape self.fft_window to [1, 1, filter_length] for broadcasting
        windowed_forward_basis = self.forward_basis * self.fft_window.view(1, 1, -1)
        #print("forward_basis stats:", self.forward_basis.min(), self.forward_basis.max())
        assert self.forward_basis.min() is not None


        forward_transform = F.conv1d(
            input_data,
            windowed_forward_basis, # Use the windowed basis here
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # we experienced strong issues when argument inside sqrt was too close from zero
        # we added an epsilon
        eps=1e-6
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + eps)
        
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase
    



def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0,mel_trainable=True,fft_learnable=True,learn_win=True):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFTLEARN(filter_length, hop_length, win_length,learn_fft=fft_learnable,learn_window=learn_win)
        self.mel_trainable=mel_trainable
        self.fft_trainable=fft_learnable
        
        mel_basis_np = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)

        ### mel_basis is now learnable
        if self.mel_trainable:
           
            self.register_buffer('initial_mel_basis', torch.from_numpy(mel_basis_np).float())
            self.mel_basis = nn.Parameter(self.initial_mel_basis.clone(), requires_grad=True)
        
        
        else:
            mel_basis = torch.from_numpy(mel_basis_np).float()
            self.register_buffer('mel_basis', mel_basis)
            self.initial_mel_basis = None # Not applicable if not trainable
       
        

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """

        ##### COMMENTED BECAUSE OUR DATA HAS SUCH VALUES ########
        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)

        # V1 stft is calculated with transform
        #magnitudes, phases = self.stft_fn.transform(y)
        
        # V2 stft is calculated via forward
        magnitudes, phases = self.stft_fn(y)

        #magnitudes = magnitudes.data # was breaking the graph
        magnitudes = magnitudes.clone()
        
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
    
    # getters
    def get_mel_basis(self):
        return self.mel_basis
    
    def get_stft(self):
        return self.stft_fn.get_fft_basis()
    
    def get_window(self):
        return self.stft_fn.get_window_init()
    
    # it's important to ensure variations of mel filter to be constrained during learning
    def enforce_mel_basis_constraints(self, delta=0.01):
        if self.mel_trainable and self.initial_mel_basis is not None:
            lower_bound = self.initial_mel_basis - delta
            upper_bound = self.initial_mel_basis + delta
            with torch.no_grad():
                self.mel_basis.copy_(torch.clamp(self.mel_basis, min=lower_bound, max=upper_bound))
    
    # learnable spectrogramm
    def forward(self,y):
        return self.mel_spectrogram(y)
    

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes,pretrained=True,head_trainable=True):
        super().__init__()

        # Load EfficientNet with pretrained weights
        if pretrained:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.model=efficientnet_b0()
        # Replace first conv to accept 1-channel input
        old_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        self.model.features[0][0] = new_conv

       
        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  
            #nn.BatchNorm1d(in_features),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes)
        )
        #self.model.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        # Ensure input has 4 dimensions (batch_size, channels, height, width)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension (assuming batch is already there)

        return self.model(x)   
    

class BirdNetFFT(nn.Module):
    def __init__(self, num_classes,pretrained,mel_trainable=True,efficient_trainable=True,num_features=[7,8],
                 fft_t=True,win_l=True,classifer_train=True):
        super().__init__()

        self.spectro_layer=TacotronSTFT(mel_trainable=mel_trainable,fft_learnable=fft_t,learn_win=win_l)

        self.model = CustomEfficientNet(num_classes=num_classes,pretrained=pretrained)

        '''
        Efficient net can be loaded in various configurations:
        
        '''
        # == transfer learning: pretrained + upper layers trainable
        if efficient_trainable and pretrained:
            for param in self.model.model.parameters():
                param.requires_grad = False

            assert type(num_features) is list, "features must be a list!!"
            for feature in num_features:
                
                for param in self.model.model.features[feature].parameters():
                    param.requires_grad = True

        # === No pretrained weights loaded: we train all layers ====
        elif efficient_trainable and not pretrained:
            for param in self.model.model.parameters():
                param.requires_grad = True

        # ===== INFERENCE MODE ======
        elif not efficient_trainable:
            
            for param in self.model.model.parameters():
                param.requires_grad = False
        else:
            raise ValueError('Efficient net configuration not relevant')

        # ===  classifier head ===
        if classifer_train:
            for param in self.model.model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.model.model.classifier.parameters():
                param.requires_grad = False

        # if fft_t:
        #     self.spectro_layer.stft_fn.forward_basis.requires_grad =True
        
            

    # === Getters that allow to inspect some signal processing layers
    def get_tachrotron_mel_basis(self):
        return self.spectro_layer.get_mel_basis()

    def get_tachrotron_stft(self):
        return self.spectro_layer.get_stft()
    
    def get_tachrotron_window(self):
        return self.spectro_layer.get_window()
    

    def forward(self, x):
        
        x=self.spectro_layer(x)
        
        return self.model(x)
    
audio_file_test=f"{audio_input_folder}/zitcis1/XC655341.ogg"
audio_file_test=f"{audio_input_folder}/zitcis1/XC124995.ogg"
audio_file_test=f"{audio_input_folder}/zitcis1/XC127906.ogg"

assert os.path.exists(audio_file_test)

audio_torch_spectro=get_audio_data(audio_raw_file=audio_file_test)



bird_model = BirdNetFFT(num_classes=182,pretrained=False,efficient_trainable=True,mel_trainable=True,
                        win_l=True,fft_t=True)

bird_model.eval()

output_bird=bird_model(audio_torch_spectro)



print('Prediction: ',torch.argmax(output_bird).detach().numpy())




    








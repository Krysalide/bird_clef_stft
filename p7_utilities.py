from typing import Any, Optional, Union, Collection
from typing_extensions import Literal
from typing import Callable, Generator, List, TypeVar, Union, Tuple, Any, Sequence
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
from librosa._cache import cache
import numpy as np
import librosa.util
import scipy
import warnings
import os

import librosa
import librosa.display
from IPython.display import Audio,display
from PIL import Image, ImageOps
from PIL import Image as pil
from pkg_resources import parse_version
if parse_version(pil.__version__)>=parse_version('10.0.0'):
    Image.ANTIALIAS=Image.LANCZOS
    
import matplotlib.cm as cm
#generates spectrograms are generated on audio clips this long
#audio clips shorter than this are ignored
audio_duration = 5.0

#generates up to this many clips per file (multiples of audio_duration)
#remainders of audio clip length / audio_duration are ignored
max_clips_per_audio_file = 1000

#image size (image_size x image_size)
image_size = 224

#processes up to this many audio files per species
max_audio_input_per_species = 1000

#setting this above 0 will cause processing to start at a folder after the first
#useful for restarting / continuing in case of running out of time / crash (folders are processed alphabetically)
start_at_folder_offset = 0

_WindowSpec = Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike]

@cache(level=10)
def get_window(
    window: _WindowSpec,
    Nx: int,
    *,
    fftbins: Optional[bool] = True,
) -> np.ndarray:
    """Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    See Also
    --------
    scipy.signal.get_window

    Notes
    -----
    This function caches at level 10.

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)
        return win

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ValueError(f"Window size mismatch: {len(window):d} != {Nx:d}")
    else:
        raise ValueError(f"Invalid window specification: {window!r}")


_STFTPad = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "reflect",
    "symmetric",
    "empty",
]
_PadModeSTFT = Union[_STFTPad, Callable[..., Any]]

@cache(level=20)
def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not librosa.util.is_positive_int(hop_length):
        raise ValueError(f"hop_length={hop_length} must be a positive integer")

    # Check audio is valid
    librosa.util.valid_audio(y)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = librosa.util.pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = librosa.util.expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if pad_mode in ("wrap", "maximum", "mean", "median", "minimum"):
            # Note: padding with a user-provided function "works", but
            # use at your own risk.
            # Since we don't pass-through kwargs here, any arguments
            # to a user-provided pad function should be encapsulated
            # by using functools.partial:
            #
            # >>> my_pad_func = functools.partial(pad_func, foo=x, bar=y)
            # >>> librosa.stft(..., pad_mode=my_pad_func)

            raise ValueError(
                f"pad_mode='{pad_mode}' is not supported by librosa.stft"
            )

        if n_fft > y.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"
            )

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0) for _ in range(y.ndim)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            # +1 here is to ensure enough samples to fill the window
            # fixes bug #1567
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = librosa.util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
                )
                y_frames_post = librosa.util.frame(
                    y_post, frame_length=n_fft, hop_length=hop_length
                )
                # How many extra frames do we have from the tail?
                extra += y_frames_post.shape[-1]
            else:
                # In this event, the first frame that touches tail padding would run off
                # the end of the padded array
                # We'll circumvent this by allocating an empty frame buffer for the tail
                # this keeps the subsequent logic simple
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)
    else:
        if n_fft > y.shape[-1]:
            raise ValueError(
                f"n_fft={n_fft} is too large for uncentered analysis of input signal of length={y.shape[-1]}"
            )

        # "Middle" of the signal starts at sample 0
        start = 0
        # We have no extra frames
        extra = 0

    fft = librosa.get_fftlib()

    if dtype is None:
        dtype = librosa.util.dtype_r2c(y.dtype)

    # Window the time series.
    y_frames = librosa.util.frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra

    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    elif not (np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[-1]):
        raise ValueError(
            f"Shape mismatch for provided output array out.shape={out.shape} and target shape={shape}"
        )
    elif not np.iscomplexobj(out):
        raise ValueError(f"output with dtype={out.dtype} is not of complex type")
    else:
        if np.allclose(shape, out.shape):
            stft_matrix = out
        else:
            stft_matrix = out[..., : shape[-1]]

    # Fill in the warm-up
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = int(
        librosa.util.MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix

def _spectrogram(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    power: float = 1,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, int]:
    """Retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    Parameters
    ----------
    y : None or np.ndarray
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """
    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        if n_fft is None:
            #raise ParameterError(f"Unable to compute spectrogram with n_fft={n_fft}")
            raise ValueError(f"Unable to compute spectrogram with n_fft={n_fft}")
        if y is None:
            raise ValueError(
                "Input signal must be provided to compute a spectrogram"
            )
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft

def custom_p7_melspectrogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a mel-scaled spectrogram.

    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.

    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.

    By default, ``power=2`` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time-series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)]
        spectrogram
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.
    **kwargs : additional keyword arguments for Mel filter bank parameters
    n_mels : int > 0 [scalar]
        number of Mel bands to generate
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of
        the mel band (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter
        by to unit l_p norm. See `librosa.util.normalize` for a full
        description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    S : np.ndarray [shape=(..., n_mels, t)]
        Mel spectrogram

    See Also
    --------
    librosa.filters.mel : Mel filter bank construction
    librosa.stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[3.837e-06, 1.451e-06, ..., 8.352e-14, 1.296e-11],
           [2.213e-05, 7.866e-06, ..., 8.532e-14, 1.329e-11],
           ...,
           [1.115e-05, 5.192e-06, ..., 3.675e-08, 2.470e-08],
           [6.473e-07, 4.402e-07, ..., 1.794e-08, 2.908e-08]],
          dtype=float32)

    Using a pre-computed power spectrogram would give the same result:

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D, sr=sr)

    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S_dB = librosa.power_to_db(S, ref=np.max)
    >>> img = librosa.display.specshow(S_dB, x_axis='time',
    ...                          y_axis='mel', sr=sr,
    ...                          fmax=8000, ax=ax)
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel-frequency spectrogram')
    """
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, **kwargs)

    melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
    return melspec

def custom_p7_spectrograms_for_audio(dirname, filename,image_output_folder, preview=False):
    # Calculate total duration of the audio file
    audio_path = os.path.join(dirname, filename)
    audio_data, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(path=audio_path)
    
    # Determine how many full segments of audio_duration fit in the audio file
    num_segments = int(total_duration // audio_duration)
    if num_segments == 0: return 0
    if num_segments > max_clips_per_audio_file: num_segments = max_clips_per_audio_file
        
    for segment in range(num_segments):

        offset_samples = int(segment * audio_duration * sr)
        end_samples = int(offset_samples + audio_duration * sr)
        
        segment_data = audio_data[offset_samples:end_samples]
        
        S = custom_p7_melspectrogram(y=segment_data, sr=sr, n_mels=128)
        S_db = librosa.amplitude_to_db(S, ref=np.max)
                
        #convert spectrogram data into directly into image (much faster than matplotlib)
        normalized_array = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
        
        #set color mapping 
        spectrogram_image = cm.magma(normalized_array)[:, :, :3]
        spectrogram_image = (spectrogram_image * 255).astype(np.uint8)
        spectrogram_image = Image.fromarray(spectrogram_image)
        
        #resize and flip (so consistent with original dataset)
        spectrogram_image = spectrogram_image.resize((image_size, image_size), Image.ANTIALIAS)
        spectrogram_image = ImageOps.flip(spectrogram_image)
        
        if preview:
            display(Audio(data=segment_data, rate=sr))
            return spectrogram_image
        
        output_folder = os.path.join(image_output_folder, os.path.basename(os.path.normpath(dirname)))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Naming files with an offset to indicate seconds offset
        base_filename = filename.replace('.ogg', '')
        output_filename = os.path.join(output_folder, f"{base_filename}_{int((segment + 1) * audio_duration):02d}.png")
        
        spectrogram_image.save(output_filename)

    return num_segments
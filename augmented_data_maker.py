import librosa
import soundfile as sf
import numpy as np
import os
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Paths
dataset_root_folder = '/home/christophe/birdclef/'
audio_input_folder = os.path.join(dataset_root_folder, "train_audio")
audio_augmented_folder = os.path.join(dataset_root_folder, "train_audio_aug")


os.makedirs(audio_augmented_folder, exist_ok=True)

# Augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    #Shift(p=0.5),
])

def augment_and_save_audio(input_path, output_path):
    audio, sample_rate = librosa.load(input_path, sr=None,duration=10)
    augmented_audio = augment(samples=audio, sample_rate=sample_rate)
    sf.write(output_path, augmented_audio, sample_rate)

# Process each label
label_count = 0
max_file_number=15
for label in os.listdir(audio_input_folder):
    label_path = os.path.join(audio_input_folder, label)
    if not os.path.isdir(label_path):
        continue

    # Create corresponding folder in augmented data
    augmented_label_path = os.path.join(audio_augmented_folder, label)
    os.makedirs(augmented_label_path, exist_ok=True)

    # Gather .ogg files
    ogg_files = [f for f in os.listdir(label_path) if f.endswith(".ogg")]
    
    # Process up to 10 files
    for i, ogg_file in enumerate(ogg_files[:max_file_number]):
        input_audio_path = os.path.join(label_path, ogg_file)
        output_audio_path = os.path.join(
            augmented_label_path, f"{os.path.splitext(ogg_file)[0]}_augmented.ogg"
        )
        try:
            augment_and_save_audio(input_audio_path, output_audio_path)
        except Exception as e:
            print(f"Failed to process {input_audio_path}: {e}")

    label_count += 1
    print(f"Processed label: {label} ({min(10, len(ogg_files))} files)")

print("Total labels parsed:", label_count)
assert label_count == 182, f"Expected 182 labels, found {label_count}"








# import librosa
# import soundfile as sf
# import numpy as np
# import os
# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# dataset_root_folder='/home/christophe/birdclef/'
# audio_input_folder = dataset_root_folder+"train_audio"
# audio_file_test=f"{audio_input_folder}/zitcis1/XC655341.ogg"

# output_path=dataset_root_folder+'augmented.ogg'
# assert os.path.exists(audio_file_test)

# audio_augmented_folder=dataset_root_folder+"train_audio_aug"


# # Define the augmentation pipeline
# augment = Compose([
#     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#     PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
#     Shift( p=0.5),
# ])

# def augment_and_save_audio(input_path, output_path):
#     # Load the audio file
#     audio, sample_rate = librosa.load(input_path, sr=None)

#     # Apply augmentations
#     augmented_audio = augment(samples=audio, sample_rate=sample_rate)

    
#     sf.write(output_path, augmented_audio, sample_rate)

# # Example usage
# input_audio_path =audio_file_test
# output_audio_path = output_path
# augment_and_save_audio(input_audio_path, output_audio_path)

# count=0
# for label in os.listdir(audio_input_folder):
#     print(label)
#     label_path = os.path.join(audio_input_folder, label)
    
#     assert os.path.exists(label_path)
#     count+=1
    

# print('total parsed',count)
# assert count==182
                
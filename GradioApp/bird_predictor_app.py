import gradio as gr
import torch
import numpy as np
import librosa
import os
import random
import mlflow
from collections import Counter
from torch.utils.data import Dataset
from model_gradio import BirdNetFFT

print('Gradio version:',gr.__version__)

def get_id_from_label(dict_label:dict,label:str):
    return dict_label[label]

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
    
bird_dataset=BirdAudioDatasetV2(root_dir="/home/christophe/birdclef/train_audio")

# Constants
FS = 32000
FIXED_LENGTH = int(5.0 * FS)  # 5 seconds
dataset_root_folder = '/home/christophe/birdclef/'
audio_input_folder = dataset_root_folder + "train_audio"


def get_audio_data(audio_raw_file):
    """Loads, pads/truncates audio to FIXED_LENGTH, and converts to Torch tensor."""
    audio_data, _ = librosa.load(audio_raw_file, sr=FS)
    if len(audio_data) < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    else:
        audio_data = audio_data[:FIXED_LENGTH]
    return torch.from_numpy(audio_data).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Gradio will run on device: ',device)

# Label mapping
label_list = sorted(os.listdir(os.path.join(dataset_root_folder, 'train_audio')))
label2id = dict(zip(label_list, list(range(len(label_list)))))
id2label = dict(zip(label2id.values(), label2id.keys()))

# Model initialization
bird_model = BirdNetFFT(num_classes=182,
                        mel_trainable=False,
                        efficient_trainable=False,
                        fft_t=False,
                        win_l=False,
                        num_features=False,
                        pretrained=True,
                        classifer_train=False).to(device)


print('entering loading of model')
experiment_name = "birdnet_training"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
except Exception as e:
    print(f"MLflow experiment setup failed: {e}. Ensure MLflow server is running or configure tracking URI.")

artifact_path = "model_bird_weights/model_weights.pth"
latest_run_id = 'b223ff8b8df84696aea632e616ae0a9f' 

try:
    local_path = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path=artifact_path)
    bird_model.load_state_dict(torch.load(local_path, map_location=device), strict=False)
    bird_model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model weights: {e}. Please ensure the MLflow run_id and artifact_path are correct and accessible.")



def predict_specific_file(audio_path,label,debug=False):
    """
    Predict the bird species and return:
    - prediction string
    - input file path (to be played)
    - one real sample path from dataset of that predicted species (≠ input file)
    """
    if not os.path.exists(audio_path):
        return "Error: Audio file not found.", None, None

    audio_tensor = get_audio_data(audio_path).to(device)

    with torch.no_grad():
        output = bird_model(audio_tensor)
    predicted_index = torch.argmax(output).item()
    predicted_label = id2label[predicted_index]
    if debug:
        print(predicted_label)

    # Filter matching samples excluding the input file
    matching_files = [
        fpath for fpath, label in bird_dataset.samples
        if label == predicted_label and os.path.abspath(fpath) != os.path.abspath(audio_path)
    ]

    # Pick one at random to avoid getting the same
    reference_sample_path = random.choice(matching_files) if matching_files else None
    assert reference_sample_path!=audio_path
    if label==predicted_label:
        prediction_text = f"✅ Espèce prédite correctement: {predicted_label}"
    else:
        prediction_text = f"😴 Oups!! le modèle a prédit: {predicted_label}"

    return prediction_text, audio_path, reference_sample_path

# def predict_specific_file(audio_path):
#     """
#     Predict the bird species and return:
#     - prediction string
#     - input file path (to be played)
#     - one real sample path from dataset of that predicted species
#     """
#     if not os.path.exists(audio_path):
#         return "Error: Audio file not found.", None, None

#     audio_tensor = get_audio_data(audio_path).to(device)

#     with torch.no_grad():
#         output = bird_model(audio_tensor)
#     predicted_index = torch.argmax(output).item()
#     predicted_label = id2label[predicted_index]

#     # Get one real file from dataset that matches predicted label
#     reference_sample_path = None
#     for fpath, label in bird_dataset.samples:
#         if label == predicted_label:
#             reference_sample_path = fpath
#             break

#     prediction_text = f"Espèce prédite: {predicted_label}"
#     return prediction_text, audio_path, reference_sample_path

DEFAULT_IMAGE_PATH='/home/christophe/birdclef/GradioApp/bird-7250976.jpg'

label_to_file = {}
for fpath, label in bird_dataset.samples:
    if label not in label_to_file:
        label_to_file[label] = fpath  # use the first found file ?

def predict_by_label(label):
    audio_path = label_to_file[label]
    return predict_specific_file(audio_path,label)

with gr.Blocks() as app:
    gr.HTML("""
    <style>
    
    </style>
    """)
    gr.Markdown("# Application de classification d'espece d'oiseaux par leur chant",elem_classes="custom-markdown")
    gr.Markdown("Selectionner une espèce et cliquez sur le bouton prédire",elem_classes="custom-markdown")

    label_selector = gr.Dropdown(
        choices=sorted(label_to_file.keys()),
        label="Selectionnez l'espece: ",
        elem_classes="custom-dropdown"
    )

    predict_button = gr.Button("Prédire",elem_classes="custom-button")

    text_output = gr.Textbox(label="Prédiction")
    audio_player_input = gr.Audio(label="Bande son utilisée pour la prédiction", interactive=False, type="filepath")
    audio_player_reference = gr.Audio(label="Bande son aléatoire de l'espèce prédite", interactive=False, type="filepath")
    image_output = gr.Image(label="", value=DEFAULT_IMAGE_PATH, height=800, width=1400, interactive=False)

    # Single button click triggers all outputs
    predict_button.click(
        fn=predict_by_label,
        inputs=label_selector,
        outputs=[text_output, audio_player_input, audio_player_reference]
    )

if __name__ == "__main__":
    app.launch(share=True)



# oldy code
# with gr.Blocks() as app:
#     gr.Markdown("# Bird Species Classifier")
#     gr.Markdown("Select an audio file from the dataset and click predict.")

   
#     label_selector = gr.Dropdown(
#         choices=sorted(label_to_file.keys()),
#         label="Select Bird Species"
#     )
    

#     with gr.Row():
#         predict_button = gr.Button("Predict")
#         play_button = gr.Button("Play Audio")

#     # Output text and image
#     text_output = gr.Textbox(label="Prediction")
#     audio_player = gr.Audio(label="Audio Preview", interactive=False, type="filepath")
#     image_output = gr.Image(label="", value=DEFAULT_IMAGE_PATH, height=800, width=1400, interactive=False)

#     predict_button.click(
#         fn=lambda label: predict_specific_file(label_to_file[label]),
#         inputs=label_selector,
#         outputs=text_output
#     )

#     play_button.click(
#         fn=lambda label: label_to_file[label],
#         inputs=label_selector,
#         outputs=audio_player
#     )




# naive version 1
# --- Gradio Interface ---
# with gr.Blocks() as app:
#     gr.Markdown("# Bird Species Classifier (Simple Demo)")
#     gr.Markdown("Click a button to get a prediction for a pre-defined bird audio file.")

#     with gr.Row():
#         # Button for audio_file_1
#         btn1 = gr.Button(f"Predict for XC655341.ogg")
#         # Button for audio_file_2
#         btn2 = gr.Button(f"Predict for XC124995.ogg")
#         # Button for audio_file_3
#         btn3 = gr.Button(f"Predict for XC127906.ogg")

#     text_output = gr.Textbox(label="Prediction")

    

#     image_output = gr.Image(label="", value=DEFAULT_IMAGE_PATH, height=800, width=1400, interactive=False)

#     # Attach click events to buttons
#     btn1.click(fn=lambda: predict_specific_file(audio_file_1), inputs=None, outputs=text_output)
#     btn2.click(fn=lambda: predict_specific_file(audio_file_2), inputs=None, outputs=text_output)
#     btn3.click(fn=lambda: predict_specific_file(audio_file_3), inputs=None, outputs=text_output)
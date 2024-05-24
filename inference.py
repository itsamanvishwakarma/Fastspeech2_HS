import sys
import os
import argparse
import torch
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
import json
import yaml
from text_preprocess_for_inference import TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor
import concurrent.futures
import numpy as np
import time

# Replace with the actual path to the HiFi-GAN directory
sys.path.append(os.path.join(os.getcwd(), "hifigan"))

from models import Generator
from env import AttrDict
from meldataset import MAX_WAV_VALUE

SAMPLING_RATE = 22050

def load_hifigan_vocoder(language, gender, device):
    vocoder_config = f"vocoder/{gender}/aryan/hifigan/config.json"
    vocoder_generator = f"vocoder/{gender}/aryan/hifigan/generator"
    with open(vocoder_config, 'r') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_fastspeech2_model(language, gender, device):
    with open(f"{language}/{gender}/model/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    current_working_directory = os.getcwd()
    feat = "model/feats_stats.npz"
    pitch = "model/pitch_stats.npz"
    energy = "model/energy_stats.npz"

    feat_path = os.path.join(current_working_directory, language, gender, feat)
    pitch_path = os.path.join(current_working_directory, language, gender, pitch)
    energy_path = os.path.join(current_working_directory, language, gender, energy)

    config["normalize_conf"]["stats_file"] = feat_path
    config["pitch_normalize_conf"]["stats_file"] = pitch_path
    config["energy_normalize_conf"]["stats_file"] = energy_path

    with open(f"{language}/{gender}/model/config.yaml", "w") as file:
        yaml.dump(config, file)

    tts_model = f"{language}/{gender}/model/model.pth"
    tts_config = f"{language}/{gender}/model/config.yaml"

    return Text2Speech(train_config=tts_config, model_file=tts_model, device=device)

def text_synthesis(language, gender, sample_text, vocoder, MAX_WAV_VALUE, device, alpha):
    with torch.no_grad():
        model = load_fastspeech2_model(language, gender, device)
        out = model(sample_text, decode_conf={"alpha": alpha})
        print("TTS Done")
        x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
        x = x.to(device)
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        return audio

def split_into_chunks(text, words_per_chunk=1500):
    words = text.split()
    chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]
    return [' '.join(chunk) for chunk in chunks]

def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech Inference")
    parser.add_argument("--language", type=str, required=True, help="Language (e.g., hindi)")
    parser.add_argument("--gender", type=str, required=True, help="Gender (e.g., female)")
    parser.add_argument("--text_file", type=str, help="Path to the text file to be synthesized")
    parser.add_argument("--results_dir", type=str, help="Directory to save results.", default="final_results")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha Parameter")

    args = parser.parse_args()

    phone_dictionary = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocoder = load_hifigan_vocoder(args.language, args.gender, device)

    if args.language == "urdu" or args.language == "punjabi":
        preprocessor = CharTextPreprocessor()
    elif args.language == "english":
        preprocessor = TTSPreprocessor()
    else:
        preprocessor = TTSDurAlignPreprocessor()

    start_time = time.time()
    audio_arr = []

    if args.text_file:
        sample_text = read_text_from_file(args.text_file)
        filename = os.path.basename(args.text_file).split(".")[0]
    else:
        raise ValueError("No text input provided. Please provide a text file using the --text_file argument.")

    result = split_into_chunks(sample_text)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for sample_text in result:
            preprocessed_text, phrases = preprocessor.preprocess(sample_text, args.language, args.gender, phone_dictionary)
            preprocessed_text = " ".join(preprocessed_text)
            audio = text_synthesis(args.language, args.gender, preprocessed_text, vocoder, MAX_WAV_VALUE, device, args.alpha)
            audio_arr.append(audio)

    result_array = np.concatenate(audio_arr, axis=0)
    result_dir = os.path.join(args.results_dir, filename)
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"{filename}.wav")
    write(output_file, SAMPLING_RATE, result_array)

    print(f"Wrote results to {output_file}")

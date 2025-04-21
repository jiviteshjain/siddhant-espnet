import numpy as np
import torch
import os
import librosa
import requests
from typing import Optional, List
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.slu_inference import Speech2Understand
from espnet2.sds.utils.utils import int2float
from scipy.io import wavfile
import requests
import os
import json
import sys

def get_vap_prediction(server_url, wav_file_path):
    """
    Send a request to the VAP server to get speaker prediction.
    
    Args:
        server_url (str): The base URL of the VAP server
        wav_file_path (str): Path to the WAV file to analyze
    
    Returns:
        dict: The server response as a dictionary
    """
    # Make sure the file exists
    if not os.path.exists(wav_file_path):
        print(f"Error: WAV file not found at {wav_file_path}")
        return {}

    # Construct the JSON data with the path_to_file key
    json_data = {'path_to_file': wav_file_path}
    
    try:
        # Send the POST request with JSON data to the server
        response = requests.post(
            f"{server_url}/get_vap_prediction", 
            json=json_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        return result    
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the server: {e}")
        return {}

class VAPModel(torch.nn.Module):
    def __init__(self, server_url: str = "http://localhost:5000"):
        super().__init__()
        self.server_url = server_url
        self.stored_audio: Optional[np.ndarray] = None
        self.stored_bin_audio: Optional[np.ndarray] = None

    def warmup(self):
        return
    
    def forward(self, speech: np.ndarray,
                sample_rate: int,
                binary: bool=False) -> Optional[np.ndarray]:

        audio_int16 = np.frombuffer(speech, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        
        if self.stored_audio is None:
            full_audio_float32 = audio_float32
            full_audio_bin = speech
        else:
            full_audio_float32 = np.concatenate([self.stored_audio, audio_float32])
            full_audio_bin = np.concatenate([self.stored_bin_audio, speech])
        print(full_audio_float32.shape)
        # assert np.max(full_audio_float32) <= 1.0, np.max(full_audio_float32)
        # assert np.min(full_audio_float32) >= -1.0, np.min(full_audio_float32)
        # full_audio_float32 = np.clip(full_audio_float32, -1.0, 1.0)
        
        # Save as WAV file
        output_file = '/home/jivitesj/projects/speech/espnet/espnet2/sds/turn_taking/vap_input2.wav'
        wavfile.write(output_file, sample_rate, full_audio_float32)
        result_dict = get_vap_prediction(self.server_url, output_file)
        x = full_audio_float32.shape[0]
        print(result_dict)
        turn_change = False
        if "speaker_change" in result_dict:
            if result_dict['speaker_change'] == 1:
                turn_change = True
        else:
            print('Error in VAP')

        if turn_change:
            print("VAP model: turn change predicted")
            self.stored_audio = None
            self.stored_bin_audio = None
            if binary:
                return full_audio_bin
            else:
                return full_audio_float32
        else:
            self.stored_audio = full_audio_float32
            self.stored_bin_audio = full_audio_bin
            return None
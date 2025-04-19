import numpy as np
import torch
import os
import librosa

from typing import Optional, List

from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.slu_inference import Speech2Understand
from espnet2.sds.utils.utils import int2float


class TalkingTurnsModel(torch.nn.Module):

    def __init__(self,
                 data_path: str,
                 device: str = "cuda",
                 target_sample_rate: int = 16000
                 ) -> None:
        
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.model = Speech2Understand(os.path.join(data_path, "config.yaml"),
                                 os.path.join(data_path, "valid.loss.ave.pth"),
                                 device=device, run_chunk=True)
        
        self.stored_audio: Optional[np.ndarray] = None
        self.stored_bin_audio: Optional[np.ndarray] = None

    def warmup(self):
        return
    
    def forward(self, speech: np.ndarray,
                sample_rate: int,
                binary: bool=False) -> Optional[np.ndarray]:

        audio_int16 = np.frombuffer(speech, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        audio_float32 = librosa.resample(
            audio_float32, orig_sr=sample_rate, target_sr=self.target_sample_rate
        )


        if self.stored_audio is None:
            full_audio_float32 = audio_float32
            full_audio_bin = speech
        else:
            full_audio_float32 = np.concatenate([self.stored_audio, audio_float32])
            full_audio_bin = np.concatenate([self.stored_bin_audio, speech])

        if full_audio_float32.shape[0] < 3840:
            print("TALKING TURNS: Short audio")
            # Consider this as "C", because we can't run the model on it.
            self.stored_audio = full_audio_float32
            self.stored_bin_audio = full_audio_bin
            return None

        prediction = self.model(full_audio_float32)[0][1][-1]
        print(f"TALKING TURNS: Prediction: {prediction}")

        if prediction == "T":
            # The input audio is from the mic, so it only contains the user's speech.
            # So this means that the user has stopped speaking, i.e., we can speak.
            self.stored_audio = None
            self.stored_bin_audio = None
            if binary:
                return full_audio_bin
            else:
                return full_audio_float32
            
        if prediction == "I":
            # Because the input audio is from the mic, it only contains the user's speech,
            # so an interruption is unexpected, because it requires two speakers simultaneously.
            # But we still consider this same as "T".
            print("TALKING TURNS: Unexpected: Interruption detected, but we consider it as 'T'.")
            self.stored_audio = None
            self.stored_bin_audio = None
            if binary:
                return full_audio_bin
            else:
                return full_audio_float32
            
        if prediction == "NA":
            # There is no speech at the end (remember we are checking the last frame),
            # so we check if we have stored audio.
            # If something is stored, that means there was a "C" predicted before, so we return
            # whatever is stored.
            if self.stored_audio is not None:
                self.stored_audio = None
                self.stored_bin_audio = None
                if binary:
                    return full_audio_bin
                else:
                    return full_audio_float32
            else:
                return None

        if prediction == "C":
            # The user is speaking, let them speak.
            # We store the audio and return None.
            self.stored_audio = full_audio_float32
            self.stored_bin_audio = full_audio_bin
            return None

        if prediction == "BC":
            # Because the input audio is from the mic, it only contains the user's speech,
            # so a backchannel is unexpected, because it requires two speakers simultaneously.
            # But we still consider this same as "C".
            print("TALKING TURNS: Unexpected: Backchannelling detected, but we consider it as 'C'.")
            self.stored_audio = full_audio_float32
            self.stored_bin_audio = full_audio_bin
            return None

        print("TALKING TURNS: Unexpected prediction, returning None.")
        return None
            

        



        

import logging
import os
import json
import re
import soundfile as sf
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedAed, FireRedAedConfig

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExpertASR")

@dataclass
class ExpertAsrConfig:
    # Model paths
    model_root: str = "fireredasr2s/pretrained_models"
    vad_model_dir: str = "FireRedVAD/AED"
    lid_model_dir: str = "FireRedLID"
    asr_type: str = "aed"
    asr_model_dir: str = "FireRedASR2-AED"
    punc_model_dir: str = "FireRedPunc"

    # VAD/AED settings - Fine-tuned for balancing speech and singing
    speech_threshold: float = 0.35
    singing_threshold: float = 0.25 # Lower threshold to catch faint singing starts/ends
    min_event_frame: int = 20
    max_event_frame: int = 2000
    min_silence_frame: int = 15
    merge_threshold_s: float = 0.8  # Aggressively merge nearby segments to avoid clipping
    padding_s: float = 0.5          # Generous padding (500ms) to ensure no characters are lost

    # ASR settings
    beam_size: int = 5
    nbest: int = 1
    use_gpu: bool = torch.cuda.is_available()

    # Output settings
    output_dir: str = "transcription_results"

class ExpertAsrSystem:
    def __init__(self, config: ExpertAsrConfig):
        self.config = config

        # Initialize sub-modules
        logger.info("Initializing models...")

        # 1. AED (Multi-label VAD)
        aed_path = os.path.join(config.model_root, config.vad_model_dir)
        aed_cfg = FireRedAedConfig(
            use_gpu=config.use_gpu,
            speech_threshold=config.speech_threshold,
            singing_threshold=config.singing_threshold,
            min_event_frame=config.min_event_frame,
            max_event_frame=config.max_event_frame,
            min_silence_frame=config.min_silence_frame
        )
        self.aed = FireRedAed.from_pretrained(aed_path, aed_cfg)

        # 2. LID
        lid_path = os.path.join(config.model_root, config.lid_model_dir)
        lid_cfg = FireRedLidConfig(use_gpu=config.use_gpu)
        self.lid = FireRedLid.from_pretrained(lid_path, lid_cfg)

        # 3. ASR
        asr_path = os.path.join(config.model_root, config.asr_model_dir)
        asr_cfg = FireRedAsr2Config(
            use_gpu=config.use_gpu,
            beam_size=config.beam_size,
            nbest=config.nbest,
            return_timestamp=True
        )
        self.asr = FireRedAsr2.from_pretrained(config.asr_type, asr_path, asr_cfg)

        # 4. Punc
        punc_path = os.path.join(config.model_root, config.punc_model_dir)
        punc_cfg = FireRedPuncConfig(use_gpu=config.use_gpu)
        self.punc = FireRedPunc.from_pretrained(punc_path, punc_cfg)

        logger.info("All models loaded successfully.")

    def _merge_segments(self, segments: List[Tuple[float, float]], dur: float) -> List[Tuple[float, float]]:
        if not segments:
            return []

        # Sort by start time
        segments.sort(key=lambda x: x[0])

        merged = []
        curr_start, curr_end = segments[0]

        for next_start, next_end in segments[1:]:
            if next_start <= curr_end + self.config.merge_threshold_s:
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

        # Apply padding and ensure within bounds
        final_segments = []
        for s, e in merged:
            s = max(0, s - self.config.padding_s)
            e = min(dur, e + self.config.padding_s)
            final_segments.append((round(s, 3), round(e, 3)))

        return final_segments

    def transcribe(self, wav_path: str) -> Dict[str, Any]:
        logger.info(f"Processing audio: {wav_path}")
        # sf.read(..., dtype="float32") returns normalized floats in [-1, 1]
        wav_np, sample_rate = sf.read(wav_path, dtype="float32")
        dur = wav_np.shape[0] / sample_rate

        # 1. AED Detection
        aed_result, _ = self.aed.detect(wav_path)
        speech_segs = aed_result["event2timestamps"].get("speech", [])
        singing_segs = aed_result["event2timestamps"].get("singing", [])

        # Union of speech and singing
        content_segs = self._merge_segments(speech_segs + singing_segs, dur)
        logger.info(f"Detected {len(content_segs)} content segments after merging and padding.")

        results = []
        for i, (start_s, end_s) in enumerate(content_segs):
            # Prepare segment for ASR
            start_sample = int(start_s * sample_rate)
            end_sample = int(end_s * sample_rate)
            wav_segment = wav_np[start_sample:end_sample]

            # FireRedASR models generally expect 16k 16-bit mono PCM.
            # If the internal models expect int16, we should convert here,
            # but usually they expect float32 normalized or they handle conversion.
            # Based on the system's own `fireredasr2system.py`, it uses int16.
            # Let's convert back to int16 for the specific model calls if needed,
            # but keep float32 for overall processing.
            wav_segment_int16 = (wav_segment * 32767).astype(np.int16)

            uttid = f"seg_{i:04d}_s{int(start_s*1000)}_e{int(end_s*1000)}"

            # 2. ASR Transcription
            asr_res = self.asr.transcribe([uttid], [(sample_rate, wav_segment_int16)])[0]
            text = asr_res.get("text", "").strip()
            if not text or re.search(r"(<blank>)|(<sil>)", text):
                continue

            # 3. LID Detection
            lid_res = self.lid.process([uttid], [(sample_rate, wav_segment_int16)])[0]

            # 4. Punctuation
            punc_res = self.punc.process_with_timestamp([asr_res.get("timestamp", [])], [uttid])[0]
            # Join all punc_sentences into one text for the segment
            text_with_punc = "".join([s["punc_text"] for s in punc_res["punc_sentences"]])

            # Format segment result
            # Check if this segment overlaps significantly with a singing event
            is_singing = False
            for s_start, s_end in singing_segs:
                # Calculate overlap
                overlap_start = max(start_s, s_start)
                overlap_end = min(end_s, s_end)
                if overlap_end > overlap_start:
                    overlap_dur = overlap_end - overlap_start
                    # If at least 30% of the segment is singing, or at least 500ms is singing
                    if overlap_dur / (end_s - start_s) > 0.3 or overlap_dur > 0.5:
                        is_singing = True
                        break

            segment_data = {
                "start_s": start_s,
                "end_s": end_s,
                "text": text_with_punc,
                "confidence": asr_res.get("confidence", 0),
                "lang": lid_res.get("lang", "unknown"),
                "is_singing": is_singing
            }
            results.append(segment_data)
            logger.info(f"[{start_s:07.2f} - {end_s:07.2f}] {'🎤' if segment_data['is_singing'] else '  '} {segment_data['text']}")

        full_text = " ".join([r["text"] for r in results])

        return {
            "wav_path": wav_path,
            "duration": dur,
            "segments": results,
            "full_text": full_text
        }

import os
import argparse
import logging
from expert_asr_system import ExpertAsrSystem, ExpertAsrConfig
from exporters import Exporter

def main():
    parser = argparse.ArgumentParser(description="FireRedASR Expert Transcription System")
    parser.add_argument("--wav_path", type=str, default="test2026.wav", help="Path to the audio file")
    parser.add_argument("--model_root", type=str, default="fireredasr2s/pretrained_models", help="Root directory for models")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--speech_threshold", type=float, default=0.35, help="VAD speech threshold")
    parser.add_argument("--singing_threshold", type=float, default=0.25, help="VAD singing threshold")

    args = parser.parse_args()

    # Custom config
    config = ExpertAsrConfig(
        model_root=args.model_root,
        output_dir=args.output_dir,
        speech_threshold=args.speech_threshold,
        singing_threshold=args.singing_threshold
    )

    # Check if audio exists
    if not os.path.exists(args.wav_path):
        print(f"Error: Audio file not found at {args.wav_path}")
        return

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run transcription
    system = ExpertAsrSystem(config)
    result = system.transcribe(args.wav_path)

    # Export results
    base_name = os.path.splitext(os.path.basename(args.wav_path))[0]
    Exporter.to_txt(result, os.path.join(args.output_dir, f"{base_name}.txt"))
    Exporter.to_srt(result, os.path.join(args.output_dir, f"{base_name}.srt"))
    Exporter.to_json(result, os.path.join(args.output_dir, f"{base_name}.json"))

    print(f"\nTranscription complete! Results saved in '{args.output_dir}'")
    print(f"- {base_name}.txt")
    print(f"- {base_name}.srt")
    print(f"- {base_name}.json")

if __name__ == "__main__":
    main()

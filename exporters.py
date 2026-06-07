import json
import datetime

class Exporter:
    @staticmethod
    def to_txt(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for seg in data["segments"]:
                tag = "🎤(歌唱)" if seg.get("is_singing") else ""
                f.write(f"[{Exporter.format_time(seg['start_s'])} - {Exporter.format_time(seg['end_s'])}] {tag} {seg['text']}\n")

    @staticmethod
    def to_srt(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for i, seg in enumerate(data["segments"]):
                f.write(f"{i + 1}\n")
                start = Exporter.format_time_srt(seg["start_s"])
                end = Exporter.format_time_srt(seg["end_s"])
                f.write(f"{start} --> {end}\n")
                tag = "🎤(歌唱) " if seg.get("is_singing") else ""
                f.write(f"{tag}{seg['text']}\n\n")

    @staticmethod
    def to_json(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def format_time(seconds):
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_int = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{seconds_int:02}.{milliseconds:03}"

    @staticmethod
    def format_time_srt(seconds):
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_int = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

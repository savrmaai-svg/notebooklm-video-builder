from moviepy.editor import AudioFileClip


def get_audio_duration(audio_path):
    with AudioFileClip(str(audio_path)) as audio:
        return float(audio.duration)


def get_target_duration(audio_path, max_seconds):
    duration = get_audio_duration(audio_path)
    return min(duration, max_seconds)


from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
VIDEO_DIR = INPUT_DIR / "videos"
AUDIO_FILE = INPUT_DIR / "audio.mp3"
TRANSCRIPT_FILE = INPUT_DIR / "transcript.txt"
OUTPUT_DIR = BASE_DIR / "output"

FINAL_VIDEO = OUTPUT_DIR / "final_video.mp4"
SUBTITLE_FILE = OUTPUT_DIR / "subtitles.srt"

TARGET_SECONDS = 10 * 60
VIDEO_SIZE = (1920, 1080)
FPS = 30
CROSSFADE_SECONDS = 0.8
SUBTITLE_MAX_CHARS = 46

FONT_SIZE = 58
FONT = "Arial-Bold"
FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttf"
SUBTITLE_COLOR = "white"
SUBTITLE_STROKE_COLOR = "black"
SUBTITLE_STROKE_WIDTH = 3

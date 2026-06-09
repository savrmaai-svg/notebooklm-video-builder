# NotebookLM + Pexels Video Builder

Is project ka flow:

```text
NotebookLM Audio (MP3)
        +
Pexels Videos (MP4 files)
        |
        v
Python Code
        |
        v
Professional 10 min Video
```

## Folder setup

```text
anmtion_video_builder/
  input/
    audio.mp3
    transcript.txt       optional, Hindi subtitle text
    videos/
      pexels_1.mp4
      pexels_2.mp4
  output/
  app.py
  main.py
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Streamlit app:

```powershell
streamlit run app.py
```

Command-line version:

```powershell
python main.py
```

Final file yahan milegi:

```text
output/final_video.mp4
```

## Notes

- Audio 10 minutes se chhota hai to final video audio jitna hi banega.
- Audio 10 minutes se bada hai to first 10 minutes use honge.
- Videos kam hon to code unhe repeat karke duration fill karega.
- `input/transcript.txt` mein Hindi text daalne par subtitles us text se banenge.
- Streamlit Cloud deployment mein Hindi subtitles ke liye transcript text paste karo.
- Local machine par auto subtitles chahiye to `pip install faster-whisper` karke empty transcript ke saath run kar sakte ho.

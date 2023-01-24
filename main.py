

import stable_whisper
from moviepy.editor import *
import os
import whisper
import sys
import time
from translate import Translator

# Calculate execution time of the script
startTime = time.time()


def annotate(clip, txt, txt_color='white', fontsize=20, font='Comic-Sans-MS'):
    """ Writes a text at the bottom of the clip. """
    txtclip = TextClip(txt, fontsize=fontsize, font=font,
                       color=txt_color, method="caption", size=(clip.w, None))
    cvc = CompositeVideoClip([clip, txtclip.set_pos(('center', 'bottom'))])
    return cvc.set_duration(clip.duration)


# If file not exist then exit script
if not os.path.exists(sys.argv[1]):
    exit()

# Open the video then write the audio to tmp audio file
video = VideoFileClip(sys.argv[1])
audio = video.audio.write_audiofile("tmp.mp3")

# Load whisper model
# If translate is required then use the small model from whisper else use the stable base model from stable-whisper
if len(sys.argv) > 4 and sys.argv[3] == "translate":
    model = whisper.load_model(sys.argv[2])
    result = model.transcribe("tmp.mp3", task="translate")
    if sys.argv[4] != "en":
        translator = Translator(
            to_lang=sys.argv[4], provider="libre", base_url="http://localhost:5000/")
else:
    model = stable_whisper.load_model(sys.argv[2])
    result = model.transcribe("tmp.mp3")

print("Transcription finished!")
# Check si il y a un trou entre le dernier text et celui actuel et si oui rajoute le clip correspondant a la duree du trou
annotatedClips = []
prevEndTime = 0
for seg in result["segments"]:
    if prevEndTime > 0 and prevEndTime != seg["start"]:
        annotatedClips.append(video.subclip(
            prevEndTime, seg["start"] if seg["start"] < video.duration else video.duration))

    if translator:
        text = translator.translate(seg["text"])
        print(seg["text"])
    else:
        text = seg["text"]
    annotatedClips.append(annotate(video.subclip(
        seg["start"], seg["end"] if seg["end"] < video.duration else video.duration), text))

    prevEndTime = seg["end"]


# Concatenate all the clip in a single video
finalVideo = concatenate_videoclips(annotatedClips)

# Delete the tmp.mp4 file if already exist then write the video to tmp.mp4
if os.path.exists("tmp.mp4"):
    os.remove("tmp.mp4")

finalVideo.write_videofile("tmp.mp4", audio=True)
os.remove("tmp.mp3")

endTime = time.time()

print("Execution time : " + str(endTime - startTime) + " s")

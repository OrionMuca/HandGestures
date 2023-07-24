from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# Play the MP3 file using the appropriate command, redirects to the audio player
# os.system(f"xdg-open {output_file}")  # For Linux-based systems
# os.system(f"start {output_file}")    # For Windows systems
# os.system(f"open {output_file}")     # For macOS systems


def text2speech(text, language):
    output_file = 'audio/' + text + '.mp3'

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output = gTTS(text=text, lang=language, slow=False)
        output.save(output_file)

    audio_segment = AudioSegment.from_mp3(output_file)
    play(audio_segment)




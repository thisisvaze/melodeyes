import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import azure.ai.vision as sdk
from PIL import Image
import time
from io import BytesIO
import cv2
from PIL import Image
import threading
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import cv2
from PIL import Image
import time
import openai
from keys import OPENAI_API_KEY, azure_vision_key

openai.api_key = OPENAI_API_KEY

class camera_sensor:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

    def get_camera_frame(self):
        # Create a VideoCapture object with the specified camera index

        # Check if the camera is opened successfully
        if not self.camera.isOpened():
            print("Error: Unable to open camera.")
            return None
        ret, frame = self.camera.read()
        # Capture a single frame from the camera
        time.sleep(0.2)
        # Capture a single frame from the camera
        ret, frame = self.camera.read()
        # Check if the frame was captured successfully
        if not ret:
            print("Error: Unable to read frame from camera.")
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil
    def release_camera(self):
                # Release the camera
        self.camera.release()

class musicgen_meta():
    global count
    def __init__(self) -> None:
        pass
        self.model = MusicGen.get_pretrained('melody')
        self.model.set_generation_params(duration=10)
        if count > 0 :
            self.melody, self.sr = torchaudio.load('./0.wav')
        else:
            pass
        
    def generate_audio(self, descriptions):
        if count > 0:
            wav = self.model.generate_with_chroma(descriptions, self.melody[None].expand(1, -1, -1), self.sr)
        else:
            wav = self.model.generate(descriptions) 
        for idx, one_wav in enumerate(wav):
            audio_write(f'{idx}', one_wav.cpu(), self.model.sample_rate, strategy="loudness")
        return f'0.wav'

def play_audio():
    audio = AudioSegment.from_wav("0.wav")
    play(audio)
    play(audio)

def play_audio_with_fade():
    audio = AudioSegment.from_wav("0.wav")
    audio_fade_out = audio.fade_out(300)
    audio_fade_in = audio.fade_in(300)
    combined_audio = audio_fade_out.append(audio_fade_in)

    play(combined_audio)
service_options = sdk.VisionServiceOptions("https://computer-vision-vaze.cognitiveservices.azure.com/",
                                           azure_vision_key)

def save_image(image_object, filename):
    # Save the image object to a local file
    image_object.save(filename)
    # Return the local file URL
    return f"./{filename}"

class azure_image_analysis():
    def __init__(self) -> None:
        pass
    def get_captions(self,image):
        vision_source = sdk.VisionSource(filename=save_image(image,"a.jpg"))
        # vision_source = sdk.VisionSource(
        #     url="https://learn.microsoft.com/azure/cognitive-services/computer-vision/media/quickstarts/presentation.png")
        analysis_options = sdk.ImageAnalysisOptions()
        analysis_options.features = (
            sdk.ImageAnalysisFeature.CAPTION
        )
        analysis_options.language = "en"

        analysis_options.gender_neutral_caption = True

        image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

        result = image_analyzer.analyze()
        if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

            if result.caption is not None:
                print("Caption:"+result.caption.content)
                return result.caption.content
        
        return "no caption"

def openai_api(query):
    try:
        
        system_query =  "Reply with a prompt for an AI based music generation for the image description which I send. Use these categories to describe the music in a short paragraph: instruments, moods, sounds, genres, rhythms, harmonies, melodies, tempo, emotion. Only reply the prompt in 2 lines."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #model="gpt-4",
            messages=[{"role": "system", "content": system_query},
                      {"role": "user", "content": "image description: " + query}]
        )
        #print(response)
        res = response["choices"][0]["message"]["content"].replace('\'', '')
        
        return res
    except:
        return "no gpt response"

# def local_ai_api(query):
#     gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")
#     # messages = [{"role": "user", "content": "Name 3 colors"}]
#     # gptj.chat_completion(messages)
#     system_query =  "Reply with a prompt for an AI based music generation for the image description which I send. Use these categories to describe the music in a short paragraph: instruments, moods, sounds, genres, rhythms, harmonies, melodies, tempo, emotion. Only reply the prompt in 2 lines."
#     response = gptj.chat_completion(
#             messages=[{"role": "user", "content": system_query + ".image description: " + query}]
#         ,
#         streaming=False)
#         #print(response)
#     res = response["choices"][0]["message"]["content"].replace('\'', '')
#     return res

def run():
    global count
    count = 0
    music_start_time = time.time()
    m = musicgen_meta()
    a = azure_image_analysis()
    c = camera_sensor()
    while True:
        start_time = time.time()
        captions = openai_api(a.get_captions(c.get_camera_frame()))
        print(captions)
        p = m.generate_audio(["A melody loop for" + captions])
        response_time = time.time() - start_time
        print(f"Request took {response_time:.2f} seconds")
        while True:
            if time.time()-music_start_time>20:
                music_start_time = time.time()
                threading.Thread(target=play_audio_with_fade, daemon=True).start()
                break

if __name__ == "__main__":
    run()
    #main()
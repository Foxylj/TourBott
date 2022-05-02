
import sys
import signal
import os





os.system("arecord -D plughw:2,0 -f S16_LE -r 8000 -c 2 01.wav")
for i in range(0,10):
	x = i

sys.exit(0)
'''
interrupted = False


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


def interrupt_callback():
    global interrupted
    return interrupted

if len(sys.argv) == 1:
    print("Error: need to specify model name")
    print("Usage: python demo.py your.model")
    sys.exit(-1)

model = sys.argv[1]

# capture SIGINT signal, e.g., Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

detector = snowboydecoder_arecord.HotwordDetector(model, sensitivity=0.5)
print('Listening... Press Ctrl+C to exit')

# main loop
detector.start(detected_callback=snowboydecoder_arecord.play_audio_file,
               interrupt_check=interrupt_callback,
               sleep_time=0.03)
detector.terminate()
'''
"""arecord -D plughw:2,0 -f S16_LE -r 8000 -c 2 01.wav"""
#sudo apt-get install sox
#sox 01.wav -r 8000 00.wav


#python.os

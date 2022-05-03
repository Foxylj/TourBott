from pynput import mouse
import sys
import signal
import os
import time

def on_click(x, y, button, pressed):
    if pressed:
        print('Start')
        return False
    else:
        print ('End')
        return False

def main():
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    os.system("arecord -D plughw:2,0 -f S16_LE -r 8000 -c 2 -d 3 01.wav")
    os.system("sox 01.wav -r 8000 00.wav")
    

if __name__ == "__main__":
    main()   


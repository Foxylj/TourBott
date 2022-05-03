from pynput import mouse
global status

def on_click(x, y, button, pressed):
    if pressed:
        print('pressed')
        return False
    else:
        print ('Released')
        return False

def main():
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    

if __name__ == "__main__":
    main()   


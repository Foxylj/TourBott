
# _*_ coding: utf-8 _*_

# 录音机，用于录制声卡播放的声音(内录)
# 可以录制鼠标操作，用于在开始录音时回放原先的鼠标操作

import os
#导入音频处理模块
import pyaudio
import threading
import wave
import time
from datetime import datetime
#导入控制与监控键盘和鼠标的模块
from pynput import keyboard,mouse

#录音类 
class Recorder():
   def __init__(self, chunk=1024, channels=2, rate=44100):
       self.CHUNK = chunk
       self.FORMAT = pyaudio.paInt16
       self.CHANNELS = channels
       self.RATE = rate
       self._running = True
       self._frames = []
       #录音开始时间
       self.recBegin =0
       #录音时长
       self.recTime =0  

   #获取内录设备序号,在windows操作系统上测试通过，hostAPI = 0 表明是MME设备
   def findInternalRecordingDevice(self,p):
       #要找查的设备名称中的关键字
       target = '立体声混音'
       #逐一查找声音设备  
       for i in range(p.get_device_count()):
           devInfo = p.get_device_info_by_index(i)   
           if devInfo['name'].find(target)>=0 and devInfo['hostApi'] == 0 :      
               #print('已找到内录设备,序号是 ',i)
               return i
       print('无法找到内录设备!')
       return -1

   #开始录音，开启一个新线程进行录音操作
   def start(self):
       print("正在录音...")  
       threading._start_new_thread(self.__record, ())

   #执行录音的线程函数
   def __record(self):
       self.recBegin = time.time()
       self._running = True
       self._frames = []

       p = pyaudio.PyAudio()
       #查找内录设备
       dev_idx = self.findInternalRecordingDevice(p)
       if dev_idx < 0 :            
           return
       #在打开输入流时指定输入设备
       stream = p.open(input_device_index=dev_idx,
                       format=self.FORMAT,
                       channels=self.CHANNELS,
                       rate=self.RATE,
                       input=True,
                       frames_per_buffer=self.CHUNK)
       #循环读取输入流
       while(self._running):
           data = stream.read(self.CHUNK)
           self._frames.append(data)

       #停止读取输入流  
       stream.stop_stream()
       #关闭输入流
       stream.close()
       #结束pyaudio
       p.terminate()
       return

   #停止录音
   def stop(self):
       self._running = False
       self.recTime = time.time() - self.recBegin
       print("录音已停止")       
       print('录音时间为%ds'%self.recTime)
       #以当前时间为关键字保存wav文件
       self.save("record/rec_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".wav")

   #保存到文件
   def save(self, fileName):   
       #创建pyAudio对象
       p = pyaudio.PyAudio()
       #打开用于保存数据的文件
       wf = wave.open(fileName, 'wb')
       #设置音频参数
       wf.setnchannels(self.CHANNELS)
       wf.setsampwidth(p.get_sample_size(self.FORMAT))
       wf.setframerate(self.RATE)
       #写入数据
       wf.writeframes(b''.join(self._frames))
       #关闭文件
       wf.close()
       #结束pyaudio
       p.terminate()

#鼠标宏 ,目前只记录与回放click操作
class MouseMacro():
   def __init__(self):        
       #指示是否记录鼠标事件
       self.enabled = False
       #模拟鼠标的控制器对象
       self.mouseCtrl = mouse.Controller()
       #记录鼠标点击位置的列表
       self.mouseMacroList=[]

   #开始记录鼠标宏操作
   def beginMouseMacro(self):
       print('开始记录鼠标宏')
       self.mouseMacroList=[]
       self.enabled=True

   #记录鼠标宏操作
   def recordMouse(self,event):
       print('记录鼠标事件',event)
       self.mouseMacroList.append(event)        

   #停止记录鼠标宏操作
   def endMouseMacro(self):
       self.enabled=False
       print('停止记录鼠标宏！')        

   #回放录制的鼠标宏操作
   def playMouseMacro(self):
       if len(self.mouseMacroList) > 0:
           print('回放鼠标宏:',self.mouseMacroList)
       for pos in self.mouseMacroList:
           self.mouseCtrl.position = pos
           self.mouseCtrl.click(mouse.Button.left,1)

#监控按键
def on_keyPress(key):
   try:
       #print('key {0} pressed'.format( key.char))
       
       #开始录音
       if key.char == 'a':
           #录音前回放鼠标宏
           mouseMacro.playMouseMacro()
           recorder.start()
       #停止录音
       if key.char == 's':
           recorder.stop() 

       #开始录制鼠标事件
       if key.char == '[':
           mouseMacro.beginMouseMacro()           
       #停止录制鼠标事件
       if key.char == ']':
           mouseMacro.endMouseMacro()    
       #测试回放鼠标宏
       if key.char == 'm':
           mouseMacro.playMouseMacro()

       #退出程序
       if key.char == 'x':           
           #mouse_listener.stop()将停止对鼠标的监听
           mouse_listener.stop()
           #返回 False 将使键盘对应的lisenter停止监听
           return False 
   
   except Exception as e: 
       print(e)


#监控鼠标
def on_click(x, y, button, pressed): 
   #print('{0} at {1}'.format('Pressed' if pressed else 'Released',(x, y)))

   #如果正在录制鼠标宏，记录鼠标的点击位置
   if pressed and mouseMacro.enabled:
       mouseMacro.recordMouse((x,y))    
   return True
   
if __name__ == "__main__":

   #检测当前目录下是否有record子目录
   if not os.path.exists('record'):
       os.makedirs('record')

   print("\npython 录音机 ....\n")
   print("----------------- 提示 ------------------------------------------\n") 
   print("按 a 键 开始录音,     按 s 键 停止录音 ,     按 x 键 退出程序 ") 
   print("按 [ 键 开始记录鼠标, 按 ] 键 停止记录鼠标 , 按 m 键 回放鼠标宏  \n") 
   print("-----------------------------------------------------------------\n") 

   #创建录音机对象
   recorder = Recorder() 
   #创建“鼠标宏”对象
   mouseMacro = MouseMacro()

   #开始监听鼠标与键盘
   keyboard_listener=keyboard.Listener(on_press=on_keyPress)
   mouse_listener=mouse.Listener(on_click=on_click)
   lst=[keyboard_listener,mouse_listener]
   for t in lst:
       t.start()
   for t in lst:
       t.join()  


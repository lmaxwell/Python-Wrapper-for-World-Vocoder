import soundfile as sf
import pyworld as pw
import time
import sys
import numpy as np
import librosa
x, fs = librosa.load(sys.argv[1],sr=16000)
# x, fs = librosa.load('utterance/vaiueo2d.wav', dtype=np.float64)

# 1. A convient way
f0, sp, ap = pw.wav2world(np.array(x,np.double), fs)    # use default options
#y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
print(sp.shape,ap.shape,f0.shape)
print(f0[0].dtype)

start_time=time.time()
print(type(fs),flush=True)
print("length {}, fs {}, time {}".format(len(x),fs,time.time()-start_time))
start_time=time.time()

wrapper=pw.wrapper2(fs,buffer_size=64,fft_size=1024,number_of_pointers=2)


wrapper._print()
wrapper.add(f0,f0.shape[0],np.ascontiguousarray(sp,np.double),np.ascontiguousarray(ap,np.double))
print("add")
wrapper._print()
i=0
y=np.array([])
while wrapper.synth()==1 :
    y=np.hstack([y,wrapper._get_buffer()])

print("synth")
wrapper._print()
wrapper.free()
print("free")
wrapper._print()

start_time=time.time()
r=wrapper.add(f0,f0.shape[0],np.ascontiguousarray(sp,np.double),np.ascontiguousarray(ap,np.double))
print("add")
wrapper._print()
print("r {}".format(r))
i=0
y=np.array([])
while wrapper.synth()==1 :
    y=np.hstack([y,wrapper._get_buffer()])
print("synth")
wrapper._print()

start_time=time.time()
r=wrapper.add(f0,f0.shape[0],np.ascontiguousarray(sp,np.double),np.ascontiguousarray(ap,np.double))
print("add")
wrapper._print()
print("r {}".format(r))
i=0
y=np.array([])
while wrapper.synth()==1 :
    y=np.hstack([y,wrapper._get_buffer()])
print("synth")
wrapper._print()

print("length {}, fs {}, time {}".format(len(x),fs,time.time()-start_time))
sf.write("synth.wav",y,fs)

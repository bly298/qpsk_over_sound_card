import numpy as np
import struct
import wave

def wav_file_read(path):
    f = wave.open(path, 'rb')
    params = f.getparams()
    nchannels, sample_size, sample_rate, nframes = params[:4]
    print ("%d %d %d %d\n" % (nchannels, sample_size, sample_rate, nframes))
    str_data = f.readframes(nframes)
    f.close()

    raw_data = np.frombuffer(str_data, dtype=np.int16)
    raw_data = np.reshape(raw_data, [nframes, nchannels])
    return raw_data

def wav_file_write(path, sample_rate, data):
    nchannels = 2
    sample_width = 2
    nframes = len(data)

    f = wave.open(path, 'wb')
    f.setparams((nchannels, sample_width, sample_rate, nframes, "NONE", "not compressed"))

    for i in range(nframes):
        val = int(data[i] * 5000)
        f.writeframes(struct.pack('h', val))
        f.writeframes(struct.pack('h', val))

    f.close()
    return
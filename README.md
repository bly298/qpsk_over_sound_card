# Modulation  
Symbol rate = 1kHz  
Carrier frequency = 10kHz  
Sampling frequency = 48kHz  

# Frame struct  
Frame = preamble + header + header checksume + content + frame checksume + frame extension  
Preamble = 13-bit barker code + 13-bit barker code  
Header = 2-byte sn + 2-byte type + 2-byte length  
Header checksum = crc32  
Frame checksume = crc32  
Frame extension = 16-byte arbitary dummy data  

# Components  
a. Preamble locator with carrier FO/phase estimation  
b. Energy detector  
c. Carrier-phase ML tracker  
d. Symbol-timing ML tracker  

# References
Umberto Mengali, Synchronization Techniques for Digital Receivers  

# Python lib dependencies  
pyaudio  
numba  

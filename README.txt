PyTirCam-1.0 

Description
===========
PyTirCam-1.0 is a Python model to manage thermal infrared camera data 
and includes two Python scripts: 

PyTirTran
---------
A transmission and conversion algorithm based on the theory of blackbody radiation 
focusing on 

    * camera sensor (spectral response)
    * atmospheric transmittance 
    * external optics (spectral transmittance) 

with the aim of quantifying and managing the effect of camera setting corrections 
by using the blackbody theory. 
It provides also a comparison of our spectral model with the empirical 
model implemeted in a thermal camera used to monitor Mt. Etna. 
Our spectral model and the empirical validated with respect of experimental data. 
It uses the spectral data contained in the corresponding folder.

PyTirConv
---------
A numerical method to recover the radiometric thermal data 
from compressed jpg thermal images 

Installation
============
Download the source files and data folders.

Usage
=====
- Run pyTirTran.py and pyTirConv.py with default data 
    * Make everything executable if necessary 
        chmod +x <filename.py>
    * Run 
        ./<filename.py>

- Run pyTirTran.py and pyTirConv.py with new data 
    * Replace the file path of the defalut data with the new directories 
    * Update the script inputs coherently to the new data 
    * Run 
        ./<filename.py> 

Contribute
==========
If you want to contribute to pyTirCam, follow the contribution guidelines and code of conduct. 

Credits
=======
- Matteo Cerminara [link github]
- Benedetta Calusi [link github]

License
=======



# MultiSpectral Ctrl module #
## Introduction ##
This is a server module aimed to run the AlliedVision camera with the Thorlabs FilterWheel to collect a MultiSpectral dataset.
The server is run on the RaspberryPi4 and controlled via a browser connected to the same network. 

## Installation ##
#### Requirements ####
This module was tested on Ubuntu 20.04 (PC) and on a RaspberryPi 4 running Raspberry Pi OS 32bit.  
RaspberryPi 3 wouldn't work because it lacks USB3 sockets.

The computer **Must** have USB-3 socket to connect the camera, and only using a USB-3 cable.

The server is run using python 3.7 with requirements detailed in requirements.txt.

#### Installation ####
1. Install python with pip support, preferably using Anaconda package manager.
2. Create a virtual environment and install the requirements file using:
`python -m pip install -r requirements.txt `
3. Install the Vimba software package from the AlliedVision site. The relevant version for the RaspberriPi4 is ARM.

## Usage ##
#### Test mode ####
- To use a dummy *Camera*: in the file `main.py`, when calling the function `get_alliedvision_grabber`,
 set the flag `use_dummy_alliedvision_camera` to `True`.
- To  use a dummy *FilterWheel*: in the file `main.py`, when calling the function `get_alliedvision_grabber`,
 set the flag `use_dummy_filterwheel` to `True`.
#### Take a photo ####
1. Connect the *FilterWheel* to the USB and to the electrical socket. Turn it on.
2. Connect the *AlliedVision* camera to the **USB-3 socket** using a **USB-3 cable**.
3. Make sure `use_dummy_alliedvision_camera` and `use_dummy_filterwheel` are set to `False` 
in the function `get_alliedvision_grabber` in file `main.py`.
4. Make sure the *RaspberryPi* and the client device (smartphone/computer) are connected to the **SAME NETWORK**.
5. Check the IP address of the **RaspberryPi**.
6. Run `python main.py` on the **RaspberryPi**.
7. On the **Client Device**: open the web browser and enter:
`<RaspberryPi IP address here>:8000`
8. Follow the instructions on the screen to take a picture.
9. To **Download** the image into the client device, press the link with the name of the image 
    on the bottom of the web page.
10. After each photograph, the results will be displayed on the bottom of the page.
11. A previously captured photo can be uploaded and display using _Upload a Photo_ button.

#### Photo name parse ####
`d<yyyymmdd>_h<hh>m<mm>s<ss>_<number of filters>Filters_<filter name 1>_<filter name 2>...tif`
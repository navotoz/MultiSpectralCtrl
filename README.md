# MultiSpectral Ctrl module #

## Introduction ##

This is a server module aimed to run the AlliedVision camera with the Thorlabs FilterWheel to collect a MultiSpectral
dataset. The server is run on the RaspberryPi4 and controlled via a browser connected to the same network.

## Installation ##

### Requirements ###

This module was tested on Ubuntu 20.04 (PC) and on a RaspberryPi 4 running Raspberry Pi OS 32bit.  
RaspberryPi 3 wouldn't work because it lacks USB3 sockets.

The computer **Must** have USB-3 socket to connect the camera, and only using a USB-3 cable.

The server is run using python 3.7 with requirements detailed in requirements.txt.

#### RaspberryPi setup

##### Time and Date issue

For some reason, the automatic time syncronization (NTP) does not work on the RPI.

Every time `apt-get` is used, the time and date should be manually set.

1. Open the terminal (CTRL+T)
2. `sudo sh` and the admin password (12345678)
3. `timedatectl set-time 2019-08-10`
4. `timedatectl set-time 21:45:53`

##### Setting up the connection #####

1. Add a file name `ssh` to the **boot* partition of the Pi.
1. The default hostname and password are *pi* and *raspberry* respectively.
1. Log into the Pi via ssh by typing `ssh pi@raspberrypi.local`
1. Change the relevant options by typing `sudo raspi-config`.
1. Change the password to `12345678` (one to eight).
1. Under *Network Options* (2.) change Hostname to `multispectralpi`.
1. Go to the options menu again, and enable VNC under *Interface options*.
1. Change the default resolution under *Advanced options*. It doesn't really matter what resolution is set, as long as
   not the default one.
1. Go back to the terminal.
1. Type `sudo vncpasswd -service`
1. Add the following lines to /root/.vnc/config.d/vncserver-x11:

```
SecurityTypes=VncAuth 
UserPasswdVerifier=VncAuth
```

1. Start vncserver by typing `sudo vncserver-x11-serviced`
1. Set the correct time:
   `sudo date --set '2016-04-26 18:26:00`
1. Reboot using `sudo reboot`
1. Notice that to connect you need to type `ssh pi@multispectralpi.local`

##### Setting up the Pythoh venv #####

1. Clone/Copy the *Multispectral* project.
2. In the terminal run `sudo apt-get install libatlas-base-dev`
3. In the project folder, make `venv` and enter into it.
4. Setup a venv inside `/venv`:
    - `python3 -m venv .`
5. Activate it by `source bin/activate`
6. Make a file in `pi/home/` called `run.sh` containing:

```
#! /bin/bash
source /path/to/multispectral/venv/bin/activate
cd /path/to/main
git reset --hard HEAD
git pull
python3 main.py
```

8. Make a new file on the desktop named `Multispectral.desktop` containing:

```
[Desktop Entry]
Terminal=true
Type=Application
X-KeepTerminal=true
StartupNotify=true
Exec=sudo /home/pi/run.sh
```

##### Python Installation #####

1. Activate the venv by `source path/to/venv/bin/activate`
1. Install the requirements file using:
   `pip install -r requirements.txt`
1. Make sure to install on pip3 also.
1. In the terminal run `sudo apt-get install libatlas-base-dev`
1. Install the Vimba software package from the AlliedVision site. The relevant version for the RaspberriPi4 is ARM.
    - An installation guide is available under _utils_ directory.
1. After running `[InstallDir]/Vimba_x_x/VimbaUSBTL`, go to `VimbaPython/Source`.
1. In `VimbaPython/Source`, using the pip *in the environment you use* run `pip install .`.

##### Setting the RaspberryPi to be a Wifi hotspot #####

1. Follow the link: https://www.raspberrypi.org/documentation/configuration/wireless/access-point-routed.md
1. Access the server by the address `192.168.4.1:8000`

##### Setting the server to run on startup #####

1. `sudo apt-get install xterm`
1. `mkdir -p ~/.config/autostart`
1. Open a terminal and type `cv ~/.config/autostart`
1. Ope a text editor and a new file `nano autostart-server.desktop`
1. Inside the file write and **Be sure to change the paths of the files**:

```
[Desktop Entry]
Encoding=UTF-8
Type=Application
Exec=/usr/bin/lxterm -e '/path/to/run.sh | less'
```

#### Changing the default filter names ####

Change **DEFAULT_FILTER_NAMES_DICT** in `devices/FilterWheel/__init.py`.

## Usage ##

#### Checking current IP address ####

1. Type in the terminal `ssh pi@multispectralpi.local` to connect to the Pi via SSH.
2. The password should be set to `12345678`
3. Type `ifconfig`

#### Take a photo ####

1. Connect the *FilterWheel* to the USB and to the electrical socket. Turn it on.
1. Connect the *AlliedVision* camera to the **USB-3 socket** using a **USB-3 cable**.
1. Connect any other relevant camera to the USB.
1. Make sure the *RaspberryPi* and the client device (smartphone/computer) are connected to the **SAME NETWORK**.
1. Run `python3 main.py` on the **RaspberryPi** with the venv (see above).
1. On the **Client Device**: open the web browser and enter:
   `multispectral.local:8000`
8. Follow the instructions on the screen to take a picture.
9. To **Download** the image into the client device, press the link with the name of the image on the bottom of the web
   page.
10. After each photograph, the results will be displayed on the bottom of the page.
11. A previously captured photo can be uploaded and display using _Upload a Photo_ button.

#### Photo

##### Structure

Two files are saved every time *Take a Photo* button is pressed:

###### The numpy files

The dimensions of the numpy file are:

(Filters, Number of samples, H, W)

- The filters are set by *Set number of filters to be photographed*, and by the filter names.
- The number of samples are set in the box *Number of images in each filter*.
- H, W are set by the camera.

###### The CSV filee

Containing the filter name and the camera temperature during the captures.

The position in the CSV file corresponds to the channel number on the numpy file.

For example, if the fourth (4) line in the CSV is filter 11000, than numpy file [3, :, :, :] contains this filter.

##### Naming

`yyyymmdd_h<hh>m<mm>s<ss>` with 'npy' and 'csv' suffix. 


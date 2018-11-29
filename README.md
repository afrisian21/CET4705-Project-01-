# FinalProject

Open Face.py and configure the Pins for GPIO to be the designated outputs that you want
by default its 13, 19, 26.

Installing OpenCV

$ sudo apt-get update

$ sudo apt-get upgrade

$ sudo apt-get dist-upgrade

After we need to restart the system

$ sudo reboot

After the reboot we install the necessary pkgs for  opencv

$ sudo apt-get install build-essential cmake pkg-config python-dev libgtk2.0-dev libgtk2.0 zlib1g-dev libpng-dev libjpeg-dev libtiff-dev libjasper-dev libavcodec-dev swig unzip

$ wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip

$ unzip opencv-2.4.9.zip

$ cd opencv-2.4.9

$ cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_gpu=OFF -DBUILD_opencv_ocl=OFF

$ sudo make install


Installing Python Dependencies

$ sudo apt-get install python-pip

$ sudo apt-get install python-dev

$ sudo pip install picamera

$ sudo pip install rpio (wont work with RP3)

Needs to get an alternative version from git hub

$ cd ~

$ git clone https://github.com/metachris/RPIO.git --branch v2 --single-branch

$ cd RPIO

$ sudo python setup.py install


Getting the software for recognition

$ git clone https://github.com/BurnedSpaghetti/FinalProject.git

$ cd FinalProject

We need to give the shell in the repository folder

$ sudo chmod +x install_dependencies.sh

Now we run the shell program

$ sudo ./install_dependencies.sh


Training the recognition

$ sudo python capture-positives.py

 every time you input c it will capture a picture and crop it to the face length and width
Once taken enough positives captures (6-20 is a good amount)

We run the train python script

$ sudo python train.py

From there it creates a training xml and 3 new images

Now Run 

$ sudo python Face.py




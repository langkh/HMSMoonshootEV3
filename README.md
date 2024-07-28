Install Anaconda (.pkg for Mac or .exe for Windows) 
Install Docker (Mac or Windows) and ensure it is running
Open Terminal (Mac) or Command Prompt (Windows)


Install AI Libraries by pasting this code in Terminal (Mac) / Command Prompt (Windows):
conda activate base 

OR you could open the Anaconda Powershell Prompt/any Commandline on Anaconda Navigator



pip install inference-cli
pip install opencv-python
pip install roboflow
pip install pygame
pip install numpy
pip install datetime
pip install ftd2xx
pip install rpyc==5.0.0


If you can't intall opencv or inference-cli it is likely a python dependency issue and you might want to use an earlier python version (3.11)
Linux:
  -When connecting to EV3 Brick make sure you have internet sharing /wifi on 
  -EV3 IP should be 192.XXX.XXX.XXX if it is 169.XXX then you are not sharing internet
Windows:
  -Windows +"R" - Type in “ncpa.cpl”
  -Go to your current Wifi and right-click on “Properties”
  -Click on the “Sharing” tab on the upper left corner. Make sure that you have “Allow other network users to connect through this computer’s internet connection” on.
  -To troubleshoot ip issues try pinging the ip that shows up when typing "ipconfig" on command line
Make sure Docker is on by typing "inference server start" on command line

On VScode, make sure you have the lego mindstorms ev3 extension installed
Add a device by typing in any name, and then typing the IP of the ev3 brick

open SSH terminal and type in ./rpyc_server.sh
Then you can run the code!

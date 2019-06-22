# EasyStyle
The style-transfering Telegram-bot.

This program implements the telegram-bot which transfers style online from the Style Image to the Content Image and and gives out the Styled Image.
You have to prepare two images - the Content Image and the Style Image in order to send them to the telegram-bot during user-dialog.

Before running the program the next set of libraries must be insalled:
install numpy
install pillow
install -U torch torchvision
install scipy
install pyTelegramBotAPI
install python-telegram-bot

Then do the following:
1) git clone --quiet https://github.com/yukonta/EasyStyle  
2) Unzip the file train2014_1000.zip from https://drive.google.com/drive/folders/1ibbe5wFviLDYzz-7SWPr0xd4alpn-Xgs into the catalog EasyStyle/neural_style/dataset_dir/train2014
3) Change the current catalog to EasyStyle
4) In the file EasyStyle/config.py (in the first string) set the actual proxy SOCKS4 for Telegram. The actual proxy address you can find in the file  https://proxyscrape.com/free-proxy-list (you need SOCKS4) - download the file and choose the address.

5) Run main.py

6) Find in Telegram the Telegram-bot EasyStyle 
7) Send the command /start to the Telegram-bot
8) Follow the instructions in the dialog: The Telegram-bot will ask you if you want to make wonderfull pictures. You have to press button "YES" or "NO". If you press "YES" the bot will offer to enter two images - first the content-image, then it will offer to enter the style-image. Then you have to wait a little (about 5 minutes) and the bot will give you the styled image (in which the style is superimposed on the content).





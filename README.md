# EasyStyle
The style-transfering Telegram-bot.

This program implements the telegram-bot which transfers style online from the style-image to the content-image and gives out the styled image.
The program sugests to transfer one of four available styles (style transfer is fast) or transfer your own style (it takes about 10 minutes).

For the first variant you have to prepare one image - content (on which the chosen available style will be transfered).

For the first variant you have to prepare two images - the content-image and the style-image.

These images you can send to the telegram-bot during user-dialog.

Before running the program the set of libraries from requirements.txt must be insalled.
if you use Anaconda: from Anaconda Prompt input the command (under your shell):

conda install --yes --file requirements.txt

If you use pip (without Anaconda) run the command:

pip install -r requirements.txt

Then do the following:
1) git clone --quiet https://github.com/yukonta/EasyStyle  
2) In the catalog EasyStyle download the file telegram_token.py which contains the telegram-token в формате token = '\<token\>'
3) Unzip the file train2014_1000.zip from https://drive.google.com/uc?id=1v79B0tLVHU3YUpIeqXQqNOT1ARl_Gov- into the catalog EasyStyle/neural_style/dataset_dir/train2014
4) Change the current catalog to EasyStyle
5) In the file EasyStyle/config.py (in the first string) set the actual proxy SOCKS4 for Telegram. The actual proxy address you can find in the file  https://proxyscrape.com/free-proxy-list (you need SOCKS4) - download the file and choose the address.

6) Run main.py
(If you have the error "AttributeError: module 'scipy.misc' has no attribute 'toimage' "  then make: pip install scipy==1.0.0)

7) Find in Telegram the Telegram-bot EasyStyle 
8) Send the command /start to the Telegram-bot
9) Follow the instructions in the dialog: The Telegram-bot will show 6 buttons. Four buttons - for transfering available styles, the fifth button - for transfering your own style and the last button you can press if you don't want to continue.

Then  the bot will offer to send images - one for transfering the available style or two for transfering you own style (first it offers to send the content-image, then it offers to enter the style-image). Then you have to wait - about 1 minute while transfering available style or 10 minutes while transfering you own style (the program must train the neural net).

Then the bot will give you the styled image (the style is superimposed on the content).


Also you can run the Telegram-Bot in Jupiter Notebook EasyStyle_neural_style.ipynb (it is also placed in EasyStyle github project).
For run the Telegram-Bot you have to run all sells in the notebook.
Do not foget to add the file telegram_token.py  in the catalog EasyStyle.




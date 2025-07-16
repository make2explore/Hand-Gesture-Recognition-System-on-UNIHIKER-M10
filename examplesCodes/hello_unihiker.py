# ---------------------------------- make2explore.com -------------------------------------------------------#
# Project           - Sample Hello UNIHIKER code
# Created By        - info@make2explore.com
# Last Modified     - 16/07/2025 17:36:00 @admin
# Software          - Python, JupyterLab, Libraries - NA
# Hardware          - DFRobot UNIHIKER M10 Dev Board    
# Sensors Used      - NA
# Source Repo       - github.com/make2explore
# ===========================================================================================================#
# Sample Hello UNIHIKER code
from unihiker import GUI
import time 


gui = GUI()

Text = gui.draw_text(text='Hello , Unihiker !',x=15,y=120,font_size=20)
Text.config(color='#0000FF')


while True:
    time.sleep(1)


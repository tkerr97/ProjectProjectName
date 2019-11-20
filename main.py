import tkinter as tk
import tkinter.filedialog

window = tk.Tk()

def loadFile():
    filename = tk.filedialog.askopenfilename()
    f = open(filename, 'r+')
    image = f.read()
    f.close()


window.title('PyScribe')
button = tk.Button(window, text='Load Image', command=loadFile).pack()
button = tk.Button(window, text='Load Model', command=loadFile).pack()

window.mainloop()

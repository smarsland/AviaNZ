import Tkinter as Tk
import tkFileDialog
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

# My first go. Now deprecated by the late decision to switch to QT!

class Interface:

    def __init__(self,root):

        self.root = root

        # Set up a canvas
        self.f = Figure(figsize=(5, 4), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.canvas.callbacks.connect('button_press_event', self.on_click)

        self.frame = Tk.Frame(self.root)

        # Set up menu and then buttons
        menu = Tk.Menu(self.root)
        self.root.config(menu=menu)

        filemenu = Tk.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open Wave File", command=self.openFile)
        filemenu.add_command(label="Quit", command=self.quit)

        Tk.Button(master=root, text='Classify', command=self.classify).pack(side=Tk.LEFT)
        Tk.Button(master=root, text='Play', command=self.play).pack(side=Tk.LEFT)
        Tk.Button(master=root, text='Quit', command=self.quit).pack(side=Tk.RIGHT)

        # Make life easier for now: preload a birdsong
        fp, self.t = wavfile.read('../Birdsong/more1.wav')
        #fp, self.t = wavfile.read('tril1.wav')
        #fp, self.t = wavfile.read('/Users/srmarsla/Students/Nirosha/bittern/ST0026.wav')
        self.spectrogram()

    def openFile(self):
        print "here"
        Formats = [ ('Wav file','*.wav')]
        file = tkFileDialog.askopenfile(parent=root,mode='rb',filetypes=Formats,initialdir=".",title='Choose a file')
        if file != None:
            fp, self.t = wavfile.read(file)
            file.close()
        self.spectrogram()

    def spectrogram(self):
        if self.t == None:
            openFile()

        window_width = 256
        incr = 128
        # This is the Hanning window
        hanning = 0.5*(1 - np.cos(2*np.pi*np.arange(window_width)/(window_width+1)))

        sg = np.zeros((window_width/2,np.ceil(len(self.t)/incr)))
        counter = 1

        for start in range(0,len(self.t)-window_width,incr):
            window = hanning*self.t[start:start+window_width]
            ft = fft(window)
            ft = ft*np.conj(ft)
            sg[:,counter] = np.real(ft[window_width/2:])
            counter += 1
        # Note that the last little bit (up to window_width) is lost. Can't just add it in since there are fewer points

        self.sg = 10.0*np.log10(sg)
        self.drawSpec()

    def drawSpec(self):
        a = self.f.add_subplot(211)
        a.plot(self.t)
        a.axis('off')
        a = self.f.add_subplot(212)
        a.imshow(self.sg,cmap='gray',aspect='auto')
        a.axis('off')
        self.canvas.draw()

    def on_key_event(self,event):
        print('you pressed %s' % event.key)
        key_press_handler(event, self.canvas, self.toolbar)

    def on_click(self,event):
        if event.inaxes is not None:
            print event.xdata, event.ydata
        else:
            print 'Clicked ouside axes bounds but inside plot window'

    def move(self,event):
        print event

    def play(self):
        import sounddevice as sd
        sd.play(self.t)

    def classify(self):
        self.newwin = Tk.Toplevel(self.root)
        # Which is the right thing: pass the signal or the fft, or both?
        # This is just the signal, and then recomputes the spectrogram -> wasteful
        self.classify = Classify(self.newwin,self.t[10000:12000])

    def quit(self):
        self.root.quit()
        self.root.destroy()

class Classify:
    # TO DO: Sort out inheritance and get this properly structured
    def __init__(self,root,t):
        self.root = root
        self.t = t
        self.e = None
        # Set up a canvas
        self.f = Figure(figsize=(5, 4), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.frame = Tk.Frame(self.root)

        Tk.Button(master=root, text='Play', command=self.play).pack(side=Tk.LEFT)
        Tk.Button(master=root, text='Don\'t Know',command=self.dk).pack(side=Tk.RIGHT)
        Tk.Button(master=root, text='Done', command=self.close).pack(side=Tk.RIGHT)

        Tk.Label(self.root, text="Please select species (or say if don't know)").pack(anchor=Tk.S)

        var = Tk.StringVar()
        self.species = Tk.StringVar()

        birds1 = [("Female Kiwi","KiwiF"),("Male Kiwi","KiwiM"),("Ruru","Ruru"),("Hihi","Hihi"),("Not bird","NB")]

        def ShowChoice():
            print var.get()
            self.species = str(var.get())


        for (key,val) in birds1:
            Tk.Radiobutton(self.root,text=key,padx=20,command=ShowChoice, value=val, variable=var).pack(side=Tk.LEFT)
        Tk.Radiobutton(self.root, text="Other", padx=20, command=self.OtherClicked, value="0").pack(side=Tk.LEFT)

        self.spectrogram()


    def spectrogram(self):

        window_width = 256
        incr = 128
        # This is the Hanning window
        hanning = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_width) / (window_width + 1)))

        sg = np.zeros((window_width / 2, np.ceil(len(self.t) / incr)))
        counter = 1

        for start in range(0, len(self.t) - window_width, incr):
            window = hanning * self.t[start:start + window_width]
            ft = fft(window)
            ft = ft * np.conj(ft)
            sg[:, counter] = np.real(ft[window_width / 2:])
            counter += 1
        # Note that the last little bit (up to window_width) is lost. Can't just add it in since there are fewer points

        self.sg = 10.0 * np.log10(sg)
        self.drawSpec()

    def drawSpec(self):
        a = self.f.add_subplot(211)
        a.plot(self.t)
        a.axis('off')
        a = self.f.add_subplot(212)
        a.imshow(self.sg, cmap='gray',aspect='auto')
        a.axis('off')
        self.canvas.draw()

    def close(self):
        if self.e is not None:
            self.species = self.e.get()
        print self.species
        self.root.destroy()

    def dk(self):
        self.species="dk"
        self.root.destroy()

    def play(self):
        import sounddevice as sd
        sd.play(self.t)

    def OtherClicked(self):
        birds2 = ["a","b","c","d","e","f","g","h","i"]
        self.l = Tk.Listbox(self.root)
        for item in birds2:
            self.l.insert(Tk.END,item)
        self.l.insert(Tk.END,"Other")
        self.l.pack(anchor=Tk.SW)
        self.l.bind('<<ListboxSelect>>', self.Selected)

    def Selected(self,event):
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        self.species = value
        if self.species=="Other":
            Tk.Label(self.root,text="Please enter").pack(anchor=Tk.SE)
            self.e = Tk.Entry(self.root)
            self.e.pack(anchor=Tk.SE)



root = Tk.Tk()
root.wm_title("AviaNZ Interface")
av = Interface(root)
root.mainloop()

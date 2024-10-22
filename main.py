import tkinter as tk
from BoxLabeler.ui import ObjectDetectionLabeler

def main():
    root = tk.Tk()
    app = ObjectDetectionLabeler(root)
    root.mainloop()

if __name__ == "__main__":
    main()

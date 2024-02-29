import tkinter as tk

def create_simple_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Currency GUI")

    # Create variables for checkboxes
    usdc_var = tk.IntVar()
    usdt_var = tk.IntVar()
    dai_var = tk.IntVar()

    # Create checkboxes
    usdc_check = tk.Checkbutton(root, text="USDC", variable=usdc_var)
    usdt_check = tk.Checkbutton(root, text="USDT", variable=usdt_var)
    dai_check = tk.Checkbutton(root, text="DAI", variable=dai_var)

    # Pack checkboxes
    usdc_check.pack()
    usdt_check.pack()
    dai_check.pack()

    # Create output box
    output = tk.Text(root, height=5, width=50)
    output.pack()

    # Create a button that updates the output box
    button = tk.Button(root, text="Update", command=lambda: output.insert(tk.END, f"USDC: {usdc_var.get()}, USDT: {usdt_var.get()}, DAI: {dai_var.get()}\n"))
    button.pack()

    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    create_simple_gui()
"""
CharityML — Finding Donors GUI
================================
Modern Tkinter GUI for donor prediction using machine learning.
Implements Random Forest, Logistic Regression, and Gradient Boosting.

Run: python finding_donors_gui.py
"""

import tkinter as tk
from app_gui import CharityMLApp

def main():
    """Run the CharityML Donor Predictor application."""
    root = tk.Tk()
    
    # Center window on screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1400) // 2
    y = (screen_height - 900) // 2
    root.geometry(f"1400x900+{x}+{y}")
    
    # Run the imported app
    app = CharityMLApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

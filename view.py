import tkinter as tk
from tkinter import messagebox, Frame, Label, Button, Canvas, OptionMenu, Scale, Checkbutton
from PIL import Image, ImageTk

import controller


class View(tk.Tk):
    CLASSIFICATIONS = ["high", "ideal", "low"]

    def __init__(self):
        super().__init__()
        self.title("3D Print Data Collector (MVC)")
        self.geometry("1000x700")
        self.controller = None
        self.current_display_image = None
        self._create_widgets()
        self._layout_widgets()
        self.set_ui_state('IDLE')

    def set_controller(self, controller):
        self.controller = controller
        self.start_auto_button.config(command=self.controller.toggle_autonomous_mode)
        self.print_button.config(command=self.controller.print_shape)
        self.capture_button.config(command=self.controller.capture_and_display_image)
        self.save_button.config(command=self.controller.save_final_image)
        self.sigma_slider.config(command=self.controller.on_sigma_change)
        for step, btn in self.view_buttons.items():
            btn.config(command=lambda s=step: self.controller.view_step(s))
        self.process_final_button.config(command=self.controller.process_to_final)

    def _create_widgets(self):
        self.control_frame = Frame(self, padx=10, pady=10)
        self.image_frame = Frame(self, bg='gray20', padx=10, pady=10)
        self.view_buttons_frame = Frame(self.control_frame)
        self.status_var = tk.StringVar(value="Status: Idle")
        self.status_label = Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w')
        self.image_canvas = Canvas(self.image_frame, bg='gray10', highlightthickness=0)
        self.start_auto_button = Button(self.control_frame, text="Start Autonomous Mode")
        self.print_button = Button(self.control_frame, text="Print Shape")
        self.capture_button = Button(self.control_frame, text="Capture Image")
        self.classification_var = tk.StringVar(value=controller.DEFAULT_AUTONOMOUS_CLASSIFICATION)
        self.classification_dropdown = OptionMenu(self.control_frame, self.classification_var, *self.CLASSIFICATIONS)
        self.save_button = Button(self.control_frame, text="Save Image")
        self.view_buttons = {}
        steps = ["Original", "Grayscale", "Blurred", "Find Outer Edges", "Closed Edges", "Crop to Shape"]
        for step in steps:
            self.view_buttons[step] = Button(self.view_buttons_frame, text=step)
        self.process_final_button = Button(self.view_buttons_frame, text="Process to Final")
        self.canny_sigma_var = tk.DoubleVar(value=0.4)
        self.sigma_slider = Scale(self.control_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.canny_sigma_var)
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.debug_mode_checkbox = Checkbutton(self.control_frame, text="Debug Mode", variable=self.debug_mode_var)

    def _layout_widgets(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky='ew')
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky='nw')
        self.image_frame.grid(row=0, column=1, rowspan=2, sticky='nsew')
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        Label(self.control_frame, text="Main Controls", font=("Arial", 12, "bold")).pack(anchor='w')
        self.start_auto_button.pack(fill=tk.X, pady=5)
        Label(self.control_frame, text="Manual Steps", font=("Arial", 12, "bold")).pack(anchor='w', pady=(10, 0))
        self.print_button.pack(fill=tk.X, pady=5)
        self.capture_button.pack(fill=tk.X, pady=5)
        Label(self.control_frame, text="Classification", font=("Arial", 12, "bold")).pack(anchor='w', pady=(10, 0))
        self.classification_dropdown.pack(fill=tk.X, pady=2)
        self.save_button.pack(fill=tk.X, pady=5)
        Label(self.control_frame, text="Processing Views", font=("Arial", 12, "bold")).pack(anchor='w', pady=(10, 0))
        self.debug_mode_checkbox.pack(anchor='w')
        self.view_buttons_frame.pack()

        buttons_in_frame = list(self.view_buttons.values())
        for i, btn in enumerate(buttons_in_frame):
            btn.grid(row=i // 2, column=i % 2, sticky='ew', padx=2, pady=2)

        last_row = (len(buttons_in_frame) + 1) // 2
        self.process_final_button.grid(row=last_row, column=0, columnspan=2, sticky='ew', padx=2, pady=2)

        Label(self.control_frame, text="Canny Edge Sigma", font=("Arial", 10)).pack(anchor='w', pady=(10, 0))
        self.sigma_slider.pack(fill=tk.X, pady=2)

    def update_image_display(self, pil_image):
        if pil_image is None:
            self.image_canvas.delete("all")
            return
        img = pil_image
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img)
        self.image_canvas.create_image(canvas_width / 2, canvas_height / 2, anchor='center', image=self.photo)

    def update_status(self, message):
        self.status_var.set(f"Status: {message}")
        self.update_idletasks()

    def set_ui_state(self, state):
        all_view_buttons = list(self.view_buttons.values()) + [self.process_final_button]
        states = {
            'print': self.print_button, 'capture': self.capture_button,
            'save': self.save_button, 'classify': self.classification_dropdown,
            'auto': self.start_auto_button, 'views': all_view_buttons
        }

        def configure_widgets(enabled, disabled):
            for key in enabled:
                widget = states[key]
                if isinstance(widget, list):
                    for w in widget: w.config(state=tk.NORMAL)
                else:
                    widget.config(state=tk.NORMAL)
            for key in disabled:
                widget = states[key]
                if isinstance(widget, list):
                    for w in widget: w.config(state=tk.DISABLED)
                else:
                    widget.config(state=tk.DISABLED)

        if state == 'IDLE':
            configure_widgets(enabled=['print', 'capture', 'auto'], disabled=['save', 'classify', 'views'])
            self.start_auto_button.config(text="Start Autonomous Mode")
        elif state == 'CAPTURED':
            configure_widgets(enabled=['print', 'capture', 'auto', 'views'], disabled=['save', 'classify'])
        elif state == 'PROCESSED':
            configure_widgets(enabled=['print', 'capture', 'auto', 'views', 'save', 'classify'], disabled=[])
        elif state == 'BUSY':
            configure_widgets(enabled=[], disabled=['print', 'capture', 'auto', 'views', 'save', 'classify'])
        elif state == 'AUTONOMOUS':
            configure_widgets(enabled=['auto'], disabled=['print', 'capture', 'views', 'save', 'classify'])
            self.start_auto_button.config(text="Stop Autonomous Mode")

    @staticmethod
    def show_error(title, message):
        messagebox.showerror(title, message)

    @staticmethod
    def show_info(title, message):
        messagebox.showinfo(title, message)

    @staticmethod
    def ask_ok_cancel(title, message):
        return messagebox.askokcancel(title, message)
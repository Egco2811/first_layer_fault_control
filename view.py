import tkinter as tk
from tkinter import ttk, PanedWindow, Entry, Scrollbar, messagebox, Frame, Label, Button, Canvas, OptionMenu, Scale, Checkbutton, Text, filedialog
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class View(tk.Tk):
    CLASSIFICATIONS = ["high", "ideal", "low"]
    Z_OFFSET_VALUES = [0.01, 0.025, 0.05, 0.1]

    def __init__(self):
        super().__init__()
        self.title("3D Print Data Collector")
        self.state('zoomed')
        self.controller = None
        self.current_display_image = None
        self.current_webcam_image = None
        self.classifier_plot_img = None

        self.acc_data = []
        self.val_acc_data = []
        self.loss_data = []
        self.val_loss_data = []

        self.notebook = ttk.Notebook(self)
        self.data_collection_tab = Frame(self.notebook)
        self.classification_tab = Frame(self.notebook)
        self.notebook.add(self.data_collection_tab, text="Data Collection")
        self.notebook.add(self.classification_tab, text="Classification")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.top_container = Frame(self.data_collection_tab)
        self.bottom_container = Frame(self.data_collection_tab, height=120)

        self._create_widgets()
        self._layout_widgets()
        self.set_ui_state('IDLE')
        self._update_z_slider_label(self.z_offset_slider_var.get())
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(100, lambda: self.top_pane.sash_place(0, 250))

    def set_controller(self, controller):
        self.controller = controller
        self.start_auto_button.config(command=self.controller.toggle_autonomous_mode)
        self.restart_firmware_button.config(command=self.controller.restart_firmware)
        self.auto_home_button.config(command=self.controller.auto_home)
        self.z_offset_up_button.config(command=self._on_z_adjust_up)
        self.z_offset_down_button.config(command=self._on_z_adjust_down)
        self.print_button.config(command=self.controller.print_shape)
        self.capture_button.config(command=self.controller.capture_and_display_image)
        self.save_button.config(command=self.controller.save_final_image)
        self.sigma_slider.config(command=self.controller.on_sigma_change)
        for step, btn in self.view_buttons.items():
            btn.config(command=lambda s=step: self.controller.view_step(s))
        self.process_final_button.config(command=self.controller.process_to_final)
        self.gcode_send_button.config(command=self._send_gcode_from_ui)
        self.gcode_entry.bind("<Return>", lambda event: self._send_gcode_from_ui())
        self.classifier_train_button.config(command=controller.start_network_training)
        self.select_file_button.config(command=self._select_prediction_file)
        self.predict_button.config(command=controller.predict_selected_file)
        self.stop_training_button.config(command=controller.stop_network_training)

    def _on_closing(self):
        if self.controller:
            self.controller.on_close()

    def _create_widgets(self):
        self.top_container = Frame(self.data_collection_tab)  
        self.bottom_container = Frame(self.data_collection_tab, height=120) 

        self.bottom_container.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.top_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.top_pane = PanedWindow(self.top_container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)

        self.control_frame_container = Frame(self.top_pane)
        self.control_canvas = Canvas(self.control_frame_container, highlightthickness=0)
        self.scrollbar = Scrollbar(self.control_frame_container, orient="vertical", command=self.control_canvas.yview)
        self.scrollable_inner_frame = Frame(self.control_canvas)
        self.top_pane.add(self.control_frame_container)

        self.image_frame = Frame(self.top_pane, bg='gray20')
        self.image_canvas = Canvas(self.image_frame, bg='gray10', highlightthickness=0)
        self.webcam_label = Label(self.image_frame, bg='black')
        self.top_pane.add(self.image_frame)

        self.terminal_frame = Frame(self.bottom_container, padx=5, pady=5)
        self.status_text = Text(self.terminal_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.gcode_input_frame = Frame(self.terminal_frame)
        self.gcode_entry = Entry(self.gcode_input_frame)
        self.gcode_send_button = Button(self.gcode_input_frame, text="Send")

        parent = self.scrollable_inner_frame
        
        self.printer_controls_frame = Frame(parent)
        self.manual_steps_frame = Frame(parent)
        self.classification_frame = Frame(parent)
        self.processing_views_frame = Frame(parent)

        self.start_auto_button = Button(self.printer_controls_frame, text="Start Autonomous Mode")
        self.restart_firmware_button = Button(self.printer_controls_frame, text="Restart Firmware")
        self.auto_home_button = Button(self.printer_controls_frame, text="Auto Home All Axes")
        self.z_offset_frame = Frame(self.printer_controls_frame, borderwidth=1, relief=tk.RIDGE)
        self.z_offset_label = Label(self.z_offset_frame, text="Set Z Offset", font=("Arial", 10, "bold"))
        self.z_offset_slider_var = tk.IntVar(value=2)
        self.z_offset_slider = Scale(self.z_offset_frame, from_=0, to=len(self.Z_OFFSET_VALUES)-1,
                                     orient=tk.HORIZONTAL, variable=self.z_offset_slider_var,
                                     showvalue=0, command=self._update_z_slider_label)
        self.z_offset_slider_label_var = tk.StringVar()
        self.z_offset_slider_label = Label(self.z_offset_frame, textvariable=self.z_offset_slider_label_var)
        self.z_offset_up_button = Button(self.z_offset_frame, text="Up")
        self.z_offset_down_button = Button(self.z_offset_frame, text="Down")

        self.print_button = Button(self.manual_steps_frame, text="Print Shape")
        self.capture_button = Button(self.manual_steps_frame, text="Capture Image")

        self.classification_var = tk.StringVar(value=self.CLASSIFICATIONS[1])
        self.classification_dropdown = OptionMenu(self.classification_frame, self.classification_var, *self.CLASSIFICATIONS)
        self.save_button = Button(self.classification_frame, text="Save Image")

        self.debug_mode_var = tk.BooleanVar(value=False)
        self.debug_mode_checkbox = Checkbutton(self.processing_views_frame, text="Debug Mode", variable=self.debug_mode_var)
        self.view_buttons_frame = Frame(self.processing_views_frame)
        self.view_buttons = {}
        steps = ["Original", "Grayscale", "Blurred", "Find Outer Edges", "Closed Edges", "Crop to Shape"]
        for step in steps:
            self.view_buttons[step] = Button(self.view_buttons_frame, text=step)
        self.process_final_button = Button(self.processing_views_frame, text="Process to Final")
        self.canny_sigma_var = tk.DoubleVar(value=0.4)
        self.sigma_slider_label = Label(self.processing_views_frame, text="Canny Edge Sigma", font=("Arial", 10))
        self.sigma_slider = Scale(self.processing_views_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.canny_sigma_var)

        self.training_controls_frame = Frame(self.classification_tab)
        self.training_controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.training_controls_inner = Frame(self.training_controls_frame)
        self.training_controls_inner.pack(anchor='center')

        Label(self.training_controls_inner, text="Epochs:").pack(side=tk.LEFT, padx=2)
        self.epochs_var = tk.IntVar(value=200)
        self.epochs_slider = Scale(
            self.training_controls_inner,
            from_=5,
            to=500,
            orient=tk.HORIZONTAL,
            variable=self.epochs_var,
            length=120,
            resolution=5  
        )
        self.epochs_slider.pack(side=tk.LEFT, padx=2)

        Label(self.training_controls_inner, text="Batch Size:").pack(side=tk.LEFT, padx=2)
        self.batch_size_var = tk.IntVar(value=8)
        self.batch_size_slider = Scale(self.training_controls_inner, from_=1, to=64, orient=tk.HORIZONTAL, variable=self.batch_size_var, length=120)
        self.batch_size_slider.pack(side=tk.LEFT, padx=2)

        Label(self.training_controls_inner, text="Learning Rate:").pack(side=tk.LEFT, padx=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_slider = Scale(self.training_controls_inner, from_=0.0001, to=0.01, resolution=0.0001, orient=tk.HORIZONTAL, variable=self.lr_var, length=120)
        self.lr_slider.pack(side=tk.LEFT, padx=2)

        self.classifier_train_button = Button(self.training_controls_inner, text="Start Network Training")
        self.classifier_train_button.pack(side=tk.LEFT, padx=10)
        self.stop_training_button = Button(self.training_controls_inner, text="Stop Training", state=tk.DISABLED)
        self.stop_training_button.pack(side=tk.LEFT, padx=10)

        self.classifier_display_frame = Frame(self.classification_tab)
        self.classifier_display_frame.pack(pady=5)
        
        self.plot_fig = Figure(figsize=(6, 3.5), dpi=100)
        self.ax_acc = self.plot_fig.add_subplot(121)
        self.ax_loss = self.plot_fig.add_subplot(122)
        self.plot_fig.tight_layout(pad=2.0)

        self.classifier_plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=self.classifier_display_frame)
        self.classifier_plot_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=5)
        
        blank_img = Image.new("RGB", (300, 300), "#E5E5E5") 
        self.selected_image_tk = ImageTk.PhotoImage(blank_img)
        self.selected_image_label = Label(self.classifier_display_frame, bg="gray90", relief=tk.SUNKEN, image=self.selected_image_tk)
        self.selected_image_label.grid(row=0, column=1, padx=10, pady=5)

        self.prediction_controls_frame = Frame(self.classification_tab)
        self.prediction_controls_frame.pack(pady=10)
        self.prediction_controls_inner = Frame(self.prediction_controls_frame)
        self.prediction_controls_inner.pack(anchor='center')

        self.select_file_button = Button(self.prediction_controls_inner, text="Select File")
        self.select_file_button.pack(side=tk.LEFT, padx=5)
        self.selected_file_label = Label(self.prediction_controls_inner, text="No file selected")
        self.selected_file_label.pack(side=tk.LEFT, padx=5)
        self.predict_button = Button(self.prediction_controls_inner, text="Predict")
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.classifier_console = Text(self.classification_tab, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.classifier_console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _layout_widgets(self):
        self.top_pane.pack(fill=tk.BOTH, expand=True)

        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.canvas_window = self.control_canvas.create_window((0, 0), window=self.scrollable_inner_frame, anchor="nw")

        def on_frame_configure(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

        def on_canvas_configure(event):
            self.control_canvas.itemconfig(self.canvas_window, width=event.width)

        self.scrollable_inner_frame.bind("<Configure>", on_frame_configure)
        self.control_canvas.bind("<Configure>", on_canvas_configure)

        self.image_frame.pack_propagate(False)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.webcam_label.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor='se')

        self.terminal_frame.pack(fill=tk.BOTH, expand=True)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.gcode_input_frame.pack(fill=tk.X)
        self.gcode_send_button.pack(side=tk.RIGHT)
        self.gcode_entry.pack(fill=tk.X, expand=True, side=tk.LEFT)

        self.scrollable_inner_frame.grid_columnconfigure(0, weight=1)

        self.printer_controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        Label(self.printer_controls_frame, text="Printer Controls", font=("Arial", 12, "bold")).pack(anchor='w')
        self.start_auto_button.pack(fill=tk.X, pady=2)
        self.restart_firmware_button.pack(fill=tk.X, pady=2)
        self.auto_home_button.pack(fill=tk.X, pady=2)
        self.z_offset_frame.pack(fill=tk.X, pady=5, expand=True)
        self.z_offset_label.pack(pady=(2,0))
        self.z_offset_slider_label.pack()
        self.z_offset_slider.pack(fill=tk.X, padx=10)
        self.z_offset_up_button.pack(fill=tk.X, padx=5, pady=(2,2))
        self.z_offset_down_button.pack(fill=tk.X, padx=5, pady=(2,5))

        self.manual_steps_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        Label(self.manual_steps_frame, text="Manual Steps", font=("Arial", 12, "bold")).pack(anchor='w')
        self.print_button.pack(fill=tk.X, pady=2)
        self.capture_button.pack(fill=tk.X, pady=2)

        self.classification_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        Label(self.classification_frame, text="Classification", font=("Arial", 12, "bold")).pack(anchor='w')
        self.classification_dropdown.pack(fill=tk.X, pady=2)
        self.save_button.pack(fill=tk.X, pady=2)

        self.processing_views_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        Label(self.processing_views_frame, text="Processing Views", font=("Arial", 12, "bold")).pack(anchor='w')
        self.debug_mode_checkbox.pack(anchor='w')
        self.view_buttons_frame.pack(fill=tk.X, pady=2)
        buttons_in_frame = list(self.view_buttons.values())
        for i, btn in enumerate(buttons_in_frame):
            btn.grid(row=i // 2, column=i % 2, sticky='ew')
        self.view_buttons_frame.grid_columnconfigure((0,1), weight=1)
        self.process_final_button.pack(fill=tk.X, pady=2)
        self.sigma_slider_label.pack(anchor='w', pady=(5,0))
        self.sigma_slider.pack(fill=tk.X)

    def _send_gcode_from_ui(self):
        command = self.gcode_entry.get()
        if command and self.controller:
            self.controller.send_gcode(command)
            self.gcode_entry.delete(0, tk.END)

    def _update_z_slider_label(self, slider_val):
        value = self.Z_OFFSET_VALUES[int(slider_val)]
        self.z_offset_slider_label_var.set(f"Step: {value:.3f}mm")

    def _get_selected_z_adjustment(self):
        index = self.z_offset_slider_var.get()
        return self.Z_OFFSET_VALUES[index]

    def _on_z_adjust_up(self):
        if self.controller:
            amount = self._get_selected_z_adjustment()
            self.controller.adjust_z(amount)

    def _on_z_adjust_down(self):
        if self.controller:
            amount = self._get_selected_z_adjustment()
            self.controller.adjust_z(-amount)

    def update_image_display(self, pil_image):
        if pil_image is None:
            self.image_canvas.delete("all")
            return
        img = pil_image
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.current_display_image = ImageTk.PhotoImage(image=img)
        self.image_canvas.create_image(canvas_width / 2, canvas_height / 2, anchor='center', image=self.current_display_image)

    def update_webcam_display(self, pil_image):
        self.current_webcam_image = ImageTk.PhotoImage(image=pil_image)
        self.webcam_label.config(image=self.current_webcam_image)

    def update_status(self, message):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def _select_prediction_file(self):
        file_path = filedialog.askopenfilename(title="Select file for prediction")
        if file_path:
            self.selected_file_label.config(text=file_path)
            self.update_selected_image(file_path)
            if self.controller:
                self.controller.set_prediction_file(file_path)

    def update_selected_image(self, image_path):
        try:
            pil_image = Image.open(image_path).resize((300, 300), Image.Resampling.LANCZOS)
            self.selected_image_tk = ImageTk.PhotoImage(pil_image)
            self.selected_image_label.config(image=self.selected_image_tk, text="")
        except Exception as e:
            self.selected_image_label.config(text="Failed to load image", image="")

    def reset_plot_data(self):
        self.acc_data.clear()
        self.val_acc_data.clear()
        self.loss_data.clear()
        self.val_loss_data.clear()

    def update_training_plot(self, epoch, logs):
        self.acc_data.append(logs.get('accuracy'))
        self.val_acc_data.append(logs.get('val_accuracy'))
        self.loss_data.append(logs.get('loss'))
        self.val_loss_data.append(logs.get('val_loss'))

        epochs_range = range(epoch + 1)

        self.ax_acc.cla()
        self.ax_acc.plot(epochs_range, self.acc_data, label='Training Accuracy')
        self.ax_acc.plot(epochs_range, self.val_acc_data, label='Validation Accuracy')
        self.ax_acc.legend(loc='lower right', fontsize='small')
        self.ax_acc.set_title('Accuracy', fontsize='medium')
        self.ax_acc.set_xlabel('Epoch', fontsize='small')
        self.ax_acc.set_ylabel('Accuracy', fontsize='small')

        self.ax_loss.cla()
        self.ax_loss.plot(epochs_range, self.loss_data, label='Training Loss')
        self.ax_loss.plot(epochs_range, self.val_loss_data, label='Validation Loss')
        self.ax_loss.legend(loc='upper right', fontsize='small')
        self.ax_loss.set_title('Loss', fontsize='medium')
        self.ax_loss.set_xlabel('Epoch', fontsize='small')
        self.ax_loss.set_ylabel('Loss', fontsize='small')

        self.classifier_plot_canvas.draw()
        self.update_idletasks()

    def update_classifier_console(self, message):
        self.classifier_console.config(state=tk.NORMAL)
        self.classifier_console.insert(tk.END, f"{message}\n")
        self.classifier_console.see(tk.END)
        self.classifier_console.config(state=tk.DISABLED)
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
            configure_widgets(enabled=['print', 'capture', 'auto', 'classify'], disabled=['save', 'views'])
            self.start_auto_button.config(text="Start Autonomous Mode")
        elif state == 'CAPTURED':
            configure_widgets(enabled=['print', 'capture', 'auto', 'views', 'classify'], disabled=['save'])
        elif state == 'PROCESSED':
            configure_widgets(enabled=['print', 'capture', 'auto', 'views', 'save', 'classify'], disabled=[])
        elif state == 'BUSY':
            configure_widgets(enabled=[], disabled=['print', 'capture', 'auto', 'views', 'save', 'classify'])
        elif state == 'AUTONOMOUS':
            configure_widgets(enabled=['auto', 'classify'], disabled=['print', 'capture', 'views', 'save'])
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

    @staticmethod
    def ask_accept_fallback_reject(title, message):
        return messagebox.askyesnocancel(title, message, icon=messagebox.QUESTION)
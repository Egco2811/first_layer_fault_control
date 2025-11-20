import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Canvas
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class View(tk.Tk):
    CLASSIFICATIONS = ["high", "ideal", "low"]
    Z_OFFSET_VALUES = [0.01, 0.025, 0.05, 0.1]
    
    COLORS = {
        "bg_dark": "#2b2b2b",
        "bg_light": "#3c3f41",
        "fg_text": "#ffffff",
        "accent": "#4a90e2",
        "success": "#6a8759",
        "danger": "#cc5c5c",
    }

    def __init__(self):
        super().__init__()
        self.title("3D Print Data Collector & Classifier")
        self.state('zoomed')
        self.configure(bg=self.COLORS['bg_dark'])
        self.controller = None
        
        self.acc_data = []
        self.val_acc_data = []
        self.loss_data = []
        self.val_loss_data = []

        self._setup_styles()
        self._create_layout()
        self._create_data_collection_ui()
        self._create_classification_ui()
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.after(100, lambda: self.set_ui_state('IDLE'))

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TFrame", background=self.COLORS['bg_dark'])
        style.configure("TLabel", background=self.COLORS['bg_dark'], foreground=self.COLORS['fg_text'], font=("Segoe UI", 10))
        style.configure("TButton", background=self.COLORS['bg_light'], foreground=self.COLORS['fg_text'], borderwidth=1)
        style.map("TButton", background=[('active', self.COLORS['accent']), ('disabled', self.COLORS['bg_dark'])])
        style.configure("Action.TButton", background=self.COLORS['accent'], font=("Segoe UI", 10, "bold"))
        style.configure("Danger.TButton", background=self.COLORS['danger'])
        style.configure("TLabelframe", background=self.COLORS['bg_dark'], bordercolor=self.COLORS['bg_light'])
        style.configure("TLabelframe.Label", background=self.COLORS['bg_dark'], foreground=self.COLORS['accent'])
        style.configure("TNotebook", background=self.COLORS['bg_dark'], borderwidth=0)
        style.configure("TNotebook.Tab", background=self.COLORS['bg_light'], foreground=self.COLORS['fg_text'], padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", self.COLORS['accent'])])
        style.configure("Horizontal.TScale", background=self.COLORS['bg_dark'], troughcolor=self.COLORS['bg_light'])
        style.configure("TCheckbutton", background=self.COLORS['bg_dark'], foreground=self.COLORS['fg_text'])

    def set_controller(self, controller):
        self.controller = controller
        self.btn_auto.config(command=self.controller.toggle_autonomous_mode)
        self.btn_restart.config(command=self.controller.restart_firmware)
        self.btn_home.config(command=self.controller.auto_home)
        self.btn_z_up.config(command=lambda: self.controller.adjust_z(self.Z_OFFSET_VALUES[self.z_slider_var.get()]))
        self.btn_z_down.config(command=lambda: self.controller.adjust_z(-self.Z_OFFSET_VALUES[self.z_slider_var.get()]))
        self.btn_print.config(command=self.controller.print_shape)
        self.btn_capture.config(command=self.controller.capture_and_display_image)
        self.btn_save.config(command=self.controller.save_final_image)
        self.btn_pipeline.config(command=self.controller.process_to_final)
        self.sigma_scale.config(command=self.controller.on_sigma_change)
        for step, btn in self.view_buttons.items():
            btn.config(command=lambda s=step: self.controller.view_step(s))
        self.btn_send_gcode.config(command=self._send_gcode)
        self.entry_gcode.bind("<Return>", lambda e: self._send_gcode())
        self.btn_train.config(command=controller.start_network_training)
        self.btn_stop_train.config(command=controller.stop_network_training)
        self.btn_select_file.config(command=self._select_prediction_file)
        self.btn_predict.config(command=controller.predict_selected_file)

    def _on_closing(self):
        if self.controller: self.controller.on_close()

    def _create_layout(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tab_collection = ttk.Frame(self.notebook)
        self.tab_classification = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_collection, text="  📷 Data Collection  ")
        self.notebook.add(self.tab_classification, text="  🧠 Classification & Training  ")

    def _create_data_collection_ui(self):
        self.tab_collection.columnconfigure(1, weight=1)
        self.tab_collection.rowconfigure(0, weight=1)
        left_panel = ttk.Frame(self.tab_collection, padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew")
        center_panel = ttk.Frame(self.tab_collection, padding=10)
        center_panel.grid(row=0, column=1, sticky="nsew")
        right_panel = ttk.Frame(self.tab_collection, padding=10)
        right_panel.grid(row=0, column=2, sticky="nsew")
        bottom_panel = ttk.Frame(self.tab_collection, padding=10)
        bottom_panel.grid(row=1, column=0, columnspan=3, sticky="ew")
        self._build_printer_controls(left_panel)
        self._build_image_view(center_panel)
        self._build_pipeline_controls(right_panel)
        self._build_console(bottom_panel)

    def _build_printer_controls(self, parent):
        lf = ttk.LabelFrame(parent, text="Printer Operations")
        lf.pack(fill=tk.X, pady=5)
        self.btn_auto = ttk.Button(lf, text="🚀 Start Autonomous", style="Action.TButton")
        self.btn_auto.pack(fill=tk.X, padx=5, pady=5)
        self.btn_home = ttk.Button(lf, text="🏠 Home All Axes")
        self.btn_home.pack(fill=tk.X, padx=5, pady=2)
        self.btn_restart = ttk.Button(lf, text="🔄 Restart Firmware", style="Danger.TButton")
        self.btn_restart.pack(fill=tk.X, padx=5, pady=2)
        
        lf_z = ttk.LabelFrame(parent, text="Z-Offset")
        lf_z.pack(fill=tk.X, pady=10)
        self.z_slider_var = tk.IntVar(value=1)
        self.lbl_z_val = ttk.Label(lf_z, text=f"Step: {self.Z_OFFSET_VALUES[1]}mm")
        self.lbl_z_val.pack(pady=(5,0))
        ttk.Scale(lf_z, from_=0, to=3, orient=tk.HORIZONTAL, variable=self.z_slider_var, command=self._update_z_label).pack(fill=tk.X, padx=5)
        f_z = ttk.Frame(lf_z)
        f_z.pack(fill=tk.X, pady=5)
        self.btn_z_up = ttk.Button(f_z, text="⬆ Up")
        self.btn_z_up.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.btn_z_down = ttk.Button(f_z, text="⬇ Down")
        self.btn_z_down.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        lf_m = ttk.LabelFrame(parent, text="Manual")
        lf_m.pack(fill=tk.X, pady=10)
        self.btn_print = ttk.Button(lf_m, text="🖨 Print Shape")
        self.btn_print.pack(fill=tk.X, padx=5, pady=2)
        self.btn_capture = ttk.Button(lf_m, text="📸 Capture")
        self.btn_capture.pack(fill=tk.X, padx=5, pady=2)

    def _build_image_view(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        container = tk.Frame(parent, bg="black", bd=2, relief=tk.SUNKEN)
        container.grid(row=0, column=0, sticky="nsew")
        self.image_canvas = Canvas(container, bg="black", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.webcam_frame = tk.Label(container, bg="black", bd=1, relief=tk.SOLID)
        self.webcam_frame.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

    def _build_pipeline_controls(self, parent):
        lf = ttk.LabelFrame(parent, text="Processing")
        lf.pack(fill=tk.X, pady=5)
        self.debug_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Debug Info", variable=self.debug_mode_var).pack(anchor="w", padx=5)
        ttk.Label(lf, text="Canny Sigma:").pack(anchor="w", padx=5, pady=(5,0))
        self.canny_sigma_var = tk.DoubleVar(value=0.4)
        self.sigma_scale = ttk.Scale(lf, from_=0.0, to=1.0, variable=self.canny_sigma_var)
        self.sigma_scale.pack(fill=tk.X, padx=5, pady=5)
        self.view_buttons = {}
        for step in ["Original", "Grayscale", "Blurred", "Find Outer Edges", "Closed Edges", "Crop to Shape"]:
            self.view_buttons[step] = ttk.Button(lf, text=step)
            self.view_buttons[step].pack(fill=tk.X, padx=5, pady=1)
        ttk.Separator(lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.btn_pipeline = ttk.Button(lf, text="✨ Full Pipeline", style="Action.TButton")
        self.btn_pipeline.pack(fill=tk.X, padx=5, pady=5)
        
        lf_s = ttk.LabelFrame(parent, text="Labeling")
        lf_s.pack(fill=tk.X, pady=10)
        self.classification_var = tk.StringVar(value=self.CLASSIFICATIONS[1])
        self.classification_dropdown = ttk.OptionMenu(lf_s, self.classification_var, self.CLASSIFICATIONS[1], *self.CLASSIFICATIONS)
        self.classification_dropdown.pack(fill=tk.X, padx=5, pady=5)
        self.btn_save = ttk.Button(lf_s, text="💾 Save", style="Action.TButton")
        self.btn_save.pack(fill=tk.X, padx=5, pady=5)

    def _build_console(self, parent):
        lf = ttk.LabelFrame(parent, text="Log")
        lf.pack(fill=tk.BOTH, expand=True)
        self.status_text = tk.Text(lf, height=5, bg=self.COLORS['bg_light'], fg=self.COLORS['fg_text'], font=("Consolas", 9), state=tk.DISABLED)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        f = ttk.Frame(lf)
        f.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.entry_gcode = ttk.Entry(f, width=20)
        self.entry_gcode.pack(fill=tk.X)
        self.btn_send_gcode = ttk.Button(f, text="Send G-Code")
        self.btn_send_gcode.pack(fill=tk.X)

    def _create_classification_ui(self):
        self.class_notebook = ttk.Notebook(self.tab_classification)
        self.class_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tab_monitor = ttk.Frame(self.class_notebook)
        self.tab_analysis = ttk.Frame(self.class_notebook)
        self.class_notebook.add(self.tab_monitor, text=" Monitor ")
        self.class_notebook.add(self.tab_analysis, text=" Analysis ")
        self._build_monitor_tab(self.tab_monitor)
        self._build_analysis_tab(self.tab_analysis)

    def _build_monitor_tab(self, parent):
        lf_conf = ttk.LabelFrame(parent, text="Config")
        lf_conf.pack(fill=tk.X, pady=5)
        f_p = ttk.Frame(lf_conf)
        f_p.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(f_p, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(f_p, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_p, text="Batch:").pack(side=tk.LEFT, padx=(15,0))
        self.batch_size_var = tk.IntVar(value=8)
        ttk.Entry(f_p, textvariable=self.batch_size_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_p, text="LR:").pack(side=tk.LEFT, padx=(15,0))
        self.lr_var = tk.DoubleVar(value=0.0001)
        ttk.Entry(f_p, textvariable=self.lr_var, width=8).pack(side=tk.LEFT, padx=5)
        self.btn_train = ttk.Button(f_p, text="▶ Start", style="Action.TButton")
        self.btn_train.pack(side=tk.LEFT, padx=20)
        self.btn_stop_train = ttk.Button(f_p, text="⏹ Stop", state=tk.DISABLED, style="Danger.TButton")
        self.btn_stop_train.pack(side=tk.LEFT)

        self.plot_fig = Figure(figsize=(5, 3), dpi=100, facecolor=self.COLORS['bg_dark'])
        self.ax_acc = self.plot_fig.add_subplot(121)
        self.ax_loss = self.plot_fig.add_subplot(122)
        self._style_plots([self.ax_acc, self.ax_loss])
        self.monitor_canvas = FigureCanvasTkAgg(self.plot_fig, master=parent)
        self.monitor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        lf_pred = ttk.LabelFrame(parent, text="Quick Test")
        lf_pred.pack(fill=tk.X, pady=5)
        f_btn = ttk.Frame(lf_pred)
        f_btn.pack(fill=tk.X, padx=5, pady=5)
        self.btn_select_file = ttk.Button(f_btn, text="📂 File")
        self.btn_select_file.pack(side=tk.LEFT)
        self.btn_predict = ttk.Button(f_btn, text="🔍 Predict", style="Action.TButton")
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        self.classifier_console = tk.Text(lf_pred, height=3, bg=self.COLORS['bg_light'], fg=self.COLORS['success'], font=("Consolas", 10), state=tk.DISABLED)
        self.classifier_console.pack(fill=tk.X, padx=5, pady=5)

    def _build_analysis_tab(self, parent):
        f_summary = ttk.Frame(parent)
        f_summary.pack(fill=tk.X, pady=10)
        self.lbl_test_acc = ttk.Label(f_summary, text="Test Accuracy: N/A", font=("Segoe UI", 16, "bold"), foreground=self.COLORS['accent'])
        self.lbl_test_acc.pack(side=tk.LEFT, padx=20)
        self.lbl_test_loss = ttk.Label(f_summary, text="Test Loss: N/A", font=("Segoe UI", 12))
        self.lbl_test_loss.pack(side=tk.LEFT, padx=20)
        self.cm_fig = Figure(figsize=(6, 5), dpi=100, facecolor=self.COLORS['bg_dark'])
        self.ax_cm = self.cm_fig.add_subplot(111)
        self.ax_cm.set_facecolor(self.COLORS['bg_dark'])
        self.ax_cm.axis('off') 
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, master=parent)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _style_plots(self, axes):
        for ax in axes:
            ax.set_facecolor(self.COLORS['bg_light'])
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

    def show_analysis_results(self, cm, classes, acc, loss):
        self.lbl_test_acc.config(text=f"Test Accuracy: {acc*100:.2f}%")
        self.lbl_test_loss.config(text=f"Test Loss: {loss:.4f}")
        self.ax_cm.clear()
        self.ax_cm.axis('on')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.ax_cm, 
                    xticklabels=classes, yticklabels=classes)
        self.ax_cm.set_title('Confusion Matrix', color='white')
        self.ax_cm.set_xlabel('Predicted', color='white')
        self.ax_cm.set_ylabel('True', color='white')
        self.ax_cm.tick_params(colors='white')
        self.cm_canvas.draw()
        self.class_notebook.select(self.tab_analysis)

    def _update_z_label(self, val):
        idx = int(float(val))
        self.lbl_z_val.config(text=f"Step: {self.Z_OFFSET_VALUES[idx]}mm")

    def _send_gcode(self):
        self.controller.send_gcode(self.entry_gcode.get())
        self.entry_gcode.delete(0, tk.END)

    def update_image_display(self, pil_image):
        if not pil_image: return
        w, h = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        if w < 10: return
        img = pil_image.copy()
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        self.current_display_image = ImageTk.PhotoImage(img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(w//2, h//2, anchor='center', image=self.current_display_image)

    def update_webcam_display(self, pil_image):
        if not pil_image: return
        pil_image.thumbnail((200, 150))
        self.current_webcam_image = ImageTk.PhotoImage(pil_image)
        self.webcam_frame.config(image=self.current_webcam_image, width=pil_image.width, height=pil_image.height)

    def update_status(self, msg):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f" >> {msg}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def update_classifier_console(self, msg):
        self.classifier_console.config(state=tk.NORMAL)
        self.classifier_console.insert(tk.END, f"{msg}\n")
        self.classifier_console.see(tk.END)
        self.classifier_console.config(state=tk.DISABLED)

    def _select_prediction_file(self):
        f = filedialog.askopenfilename()
        if f: self.controller.set_prediction_file(f)

    def reset_plot_data(self):
        self.acc_data, self.val_acc_data, self.loss_data, self.val_loss_data = [], [], [], []
        self.ax_acc.clear()
        self.ax_loss.clear()
        self._style_plots([self.ax_acc, self.ax_loss])
        self.monitor_canvas.draw()

    def update_training_plot(self, epoch, logs, msg):
        self.acc_data.append(logs.get('accuracy'))
        self.val_acc_data.append(logs.get('val_accuracy'))
        self.loss_data.append(logs.get('loss'))
        self.val_loss_data.append(logs.get('val_loss'))
        self.ax_acc.clear()
        self.ax_loss.clear()
        self._style_plots([self.ax_acc, self.ax_loss])
        self.ax_acc.plot(self.acc_data, label='Train', color='#4a90e2')
        self.ax_acc.plot(self.val_acc_data, label='Val', color='#6a8759')
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.legend()
        self.ax_loss.plot(self.loss_data, label='Train', color='#4a90e2')
        self.ax_loss.plot(self.val_loss_data, label='Val', color='#6a8759')
        self.ax_loss.set_title("Loss")
        self.ax_loss.legend()
        self.monitor_canvas.draw()
        if msg: self.update_classifier_console(msg)

    def set_ui_state(self, state):

        view_btns = list(self.view_buttons.values())
        main_controls = [self.btn_print, self.btn_capture, self.btn_auto, self.btn_pipeline, self.btn_save, self.classification_dropdown]
        all_btns = main_controls + view_btns

        disabled_all = ['disabled'] * len(all_btns)
        
        if state == 'IDLE':
            states = ['normal', 'normal', 'normal', 'disabled', 'disabled', 'normal'] + ['disabled'] * len(view_btns)
            auto_text = "🚀 Start Autonomous"
            
        elif state == 'CAPTURED':
            states = ['normal', 'normal', 'normal', 'normal', 'disabled', 'normal'] + ['normal'] * len(view_btns)
            auto_text = "🚀 Start Autonomous"
            
        elif state == 'PROCESSED':
            states = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal'] + ['normal'] * len(view_btns)
            auto_text = "🚀 Start Autonomous"
            
        elif state == 'BUSY':
            states = disabled_all
            auto_text = "⏳ Working..."
            
        elif state == 'AUTONOMOUS':
            states = ['disabled', 'disabled', 'normal', 'disabled', 'disabled', 'disabled'] + ['disabled'] * len(view_btns)
            auto_text = "🛑 Stop Auto"
        else:
            states = disabled_all
            auto_text = "Error"

        for btn, s in zip(all_btns, states):
            btn.config(state=s)
            
        self.btn_auto.config(text=auto_text)
        if state == 'AUTONOMOUS':
            self.btn_auto.config(style="Danger.TButton")
        else:
            self.btn_auto.config(style="Action.TButton")

    def show_error(self, t, m): messagebox.showerror(t, m)
    def show_info(self, t, m): messagebox.showinfo(t, m)
    def ask_ok_cancel(self, t, m): return messagebox.askokcancel(t, m)
    def ask_accept_fallback_reject(self, t, m): return messagebox.askyesnocancel(t, m)
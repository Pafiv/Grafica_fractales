import tkinter as tk
from tkinter import ttk, filedialog
import math
from PIL import Image, ImageDraw, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # aunque no se use directamente, es necesario para activar proyección 3D
import matplotlib.pyplot as plt
import numpy as np

# Aceleración con Numba
from numba import njit

@njit(fastmath=True)
def mandelbrot_escape(w, h, zoom, offset_x, offset_y, angle_deg, max_iter):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    angle = -np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for x in range(w):
        for y in range(h):
            x0 = x - w/2 - offset_x
            y0 = y - h/2 - offset_y
            xr = x0 * cos_a - y0 * sin_a
            yr = x0 * sin_a + y0 * cos_a
            cx = xr / (0.25 * w * zoom)
            cy = yr / (0.25 * h * zoom)
            zx, zy = 0.0, 0.0
            iter = 0
            while zx*zx + zy*zy < 4.0 and iter < max_iter:
                tmp = zx*zx - zy*zy + cx
                zy = 2.0*zx*zy + cy
                zx = tmp
                iter += 1
            if iter == max_iter:
                img[y, x] = (0, 0, 0)
            else:
                img[y, x] = (iter % 4 * 64, iter % 8 * 32, iter % 16 * 16)
    return img

@njit(fastmath=True)
def julia_escape(w, h, zoom, offset_x, offset_y, angle_deg, c_real, c_imag, max_iter):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    angle = -np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for x in range(w):
        for y in range(h):
            x0 = x - w/2 - offset_x
            y0 = y - h/2 - offset_y
            xr = x0 * cos_a - y0 * sin_a
            yr = x0 * sin_a + y0 * cos_a
            zx = xr / (0.25 * w * zoom)
            zy = yr / (0.25 * h * zoom)
            iter = 0
            while zx*zx + zy*zy < 4.0 and iter < max_iter:
                tmp = zx*zx - zy*zy + c_real
                zy = 2.0*zx*zy + c_imag
                zx = tmp
                iter += 1
            if iter == max_iter:
                img[y, x] = (0, 0, 0)
            else:
                img[y, x] = (iter % 4 * 64, iter % 8 * 32, iter % 16 * 16)
    return img

def show_mandelbrot_3d(self):
        # Resolución reducida para vista rápida
        w, h = 200, 150
        max_iter = self.iterations if hasattr(self, 'iterations') else 100

        if self.fractal_type == "Julia":
            # Parámetro C fijo (puedes hacerlo slider si quieres)
            data = julia_escape(w, h, self.zoom, self.offset_x, self.offset_y, self.angle, -0.7, 0.27, max_iter)
            titulo = "Julia 3D"
        else:
            data = mandelbrot_escape(w, h, self.zoom, self.offset_x, self.offset_y, self.angle, max_iter)
            titulo = "Mandelbrot 3D"

        xs = np.linspace(-2.0/self.zoom, 2.0/self.zoom, w)
        ys = np.linspace(-2.0/self.zoom, 2.0/self.zoom, h)
        X, Y = np.meshgrid(xs, ys)
        Z = np.mean(data, axis=2)  # usar intensidad como altura

        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=False)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_zlabel('Altura')
        fig.colorbar(surf, ax=ax, shrink=0.5)

        top = tk.Toplevel(self.root)
        top.title(titulo)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)



class FractalViewer:
    def __init__(self, master):
        self.root = master
        self.master = master
        master.title("Visualizador Interactivo de Fractales")
        self.width, self.height = 800, 600
        self.fractal_type = "Koch"
        self.iterations = 4
        self.zoom = 1.0
        self.angle = 0
        self.offset_x, self.offset_y = 0, 0
        self.color = (0, 0, 255)
        self.bg_color = (255, 255, 255)
        self.azim_3d = 45
        self.elev_3d = 30
        self.zoom_3d = 1.0
        self.offset_3d = [0, 0, 0]
        self.setup_gui()
        self.draw_fractal()
        self.master.configure(bg="#181828")  # Color de fondo oscuro para toda la ventana

    def setup_gui(self):
        # Título principal mejorado
        title = tk.Label(
            self.master,
            text="Visualizador Interactivo de Fractales",
            font=("Segoe UI", 26, "bold"),
            fg="#00e0ff",
            bg="#181828"
        )
        title.pack(side=tk.TOP, pady=18)

        control_frame = tk.Frame(self.master, bg="#222233")
        control_frame.pack(side=tk.LEFT, padx=14, pady=14, fill="y")

        # Etiquetas y controles con fuente moderna y color
        label_font = ("Segoe UI", 13, "bold")
        entry_font = ("Segoe UI", 12)
        tk.Label(control_frame, text="Tipo de Fractal:", font=label_font, fg="#00e0ff", bg="#222233").grid(row=0, column=0, sticky="w")
        self.fractal_var = tk.StringVar(value=self.fractal_type)
        fractal_options = ["Koch", "Sierpinski", "Mandelbrot", "Julia", "Árbol", "Sierpinski 3D", "Árbol 3D", "Koch 3D"]
        fractal_menu = ttk.Combobox(control_frame, textvariable=self.fractal_var, values=fractal_options, font=entry_font, state="readonly")
        fractal_menu.grid(row=0, column=1, pady=5)
        fractal_menu.bind("<<ComboboxSelected>>", self.change_fractal)

        tk.Label(control_frame, text="Iteraciones:", font=label_font, fg="#00e0ff", bg="#222233").grid(row=1, column=0, sticky="w")
        self.iter_slider = tk.Scale(
            control_frame, from_=1, to=7, orient=tk.HORIZONTAL, command=self.update_iterations,
            font=entry_font, bg="#181828", fg="#00e0ff", troughcolor="#333355", highlightthickness=0
        )
        self.iter_slider.set(self.iterations)
        self.iter_slider.grid(row=1, column=1, pady=5)

        # Botón Guardar Imagen
        save_btn = tk.Button(
            control_frame, text="Guardar Imagen", command=self.save_image,
            font=label_font, bg="#00e0ff", fg="#181828", activebackground="#00bfff", activeforeground="#181828"
        )
        save_btn.grid(row=2, column=0, pady=10, sticky="ew")

        # Botón Vista 3D al lado del de guardar
        self.btn_3d = tk.Button(
            control_frame,
            text="🌐 Vista 3D",
            command=self.show_mandelbrot_3d,
            font=("Segoe UI", 13, "bold"),
            bg="#00e0ff",
            fg="#181828",
            activebackground="#00bfff",
            activeforeground="#181828",
            relief="raised",
            bd=3,
            cursor="hand2"
        )
        self.btn_3d.grid(row=2, column=1, pady=10, sticky="ew")

        # Estado inicial del botón
        if self.fractal_type not in ["Mandelbrot", "Julia"]:
            self.btn_3d.config(state="disabled")

        # Botones de traslación
        move_frame = tk.LabelFrame(control_frame, text="Mover", font=label_font, fg="#00e0ff", bg="#222233", labelanchor="n")
        move_frame.grid(row=3, column=0, columnspan=2, pady=5)
        tk.Button(move_frame, text="↑", width=3, command=lambda: self.move_fractal(0, -20), font=label_font, bg="#181828", fg="#00e0ff").grid(row=0, column=1)
        tk.Button(move_frame, text="←", width=3, command=lambda: self.move_fractal(-20, 0), font=label_font, bg="#181828", fg="#00e0ff").grid(row=1, column=0)
        tk.Button(move_frame, text="→", width=3, command=lambda: self.move_fractal(20, 0), font=label_font, bg="#181828", fg="#00e0ff").grid(row=1, column=2)
        tk.Button(move_frame, text="↓", width=3, command=lambda: self.move_fractal(0, 20), font=label_font, bg="#181828", fg="#00e0ff").grid(row=2, column=1)

        # Botones de rotación
        rotate_frame = tk.LabelFrame(control_frame, text="Rotar", font=label_font, fg="#00e0ff", bg="#222233", labelanchor="n")
        rotate_frame.grid(row=4, column=0, columnspan=2, pady=5)
        tk.Button(rotate_frame, text="⟲ -15°", width=7, command=lambda: self.rotate_fractal(-15), font=label_font, bg="#181828", fg="#00e0ff").grid(row=0, column=0, padx=2)
        tk.Button(rotate_frame, text="⟳ +15°", width=7, command=lambda: self.rotate_fractal(15), font=label_font, bg="#181828", fg="#00e0ff").grid(row=0, column=1, padx=2)

        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height, bg="#181828", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        # Bindings de teclado y mouse (igual que antes)
        self.master.bind("<Up>", lambda e: self.move_fractal(0, -20))
        self.master.bind("<Down>", lambda e: self.move_fractal(0, 20))
        self.master.bind("<Left>", lambda e: self.move_fractal(-20, 0))
        self.master.bind("<Right>", lambda e: self.move_fractal(20, 0))
        self.master.bind("<Key-plus>", lambda e: self.zoom_in())
        self.master.bind("<Key-minus>", lambda e: self.zoom_out())
        self.master.bind("a", lambda e: self.rotate_fractal(-15))
        self.master.bind("d", lambda e: self.rotate_fractal(15))
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)    # Linux scroll down

        # 3D controls
        self.master.bind("<Shift-Up>", lambda e: self.move_3d(0, 0, 20))
        self.master.bind("<Shift-Down>", lambda e: self.move_3d(0, 0, -20))
        self.master.bind("<Shift-Left>", lambda e: self.move_3d(-20, 0, 0))
        self.master.bind("<Shift-Right>", lambda e: self.move_3d(20, 0, 0))
        self.master.bind("<Control-Up>", lambda e: self.change_elev(10))
        self.master.bind("<Control-Down>", lambda e: self.change_elev(-10))
        self.master.bind("<Control-Left>", lambda e: self.change_azim(-10))
        self.master.bind("<Control-Right>", lambda e: self.change_azim(10))
        self.master.bind("<Control-plus>", lambda e: self.zoom_3d_in())
        self.master.bind("<Control-minus>", lambda e: self.zoom_3d_out())

        # Cuadro de instrucciones mejorado y visible
        instructions = (
            "Controles 2D:\n"
            "  • Flechas: Mover\n"
            "  • A/D o botones: Rotar\n"
            "  • Rueda mouse: Zoom\n\n"
            "Controles 3D:\n"
            "  • Shift + Flechas: Mover\n"
            "  • Ctrl + Flechas: Rotar\n"
            "  • Ctrl + +/-: Zoom\n\n"
            "General:\n"
            "  • Selecciona el fractal en el menú\n"
            "  • Usa los botones para mover/rotar\n"
            "  • Guarda la imagen con el botón"
        )
        instr_frame = tk.LabelFrame(
            control_frame, text="Instrucciones", font=label_font,
            fg="#00e0ff", bg="#222233", labelanchor="n"
        )
        instr_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        instr_label = tk.Label(
            instr_frame, text=instructions, justify="left", anchor="w",
            font=entry_font, fg="#00e0ff", bg="#222233", wraplength=260
        )
        instr_label.pack(fill="both", expand=True, padx=6, pady=6)


    def draw_fractal(self):
        # Si hay un canvas matplotlib previo, destrúyelo
        if hasattr(self, 'mpl_canvas'):
            self.mpl_canvas.get_tk_widget().destroy()
            del self.mpl_canvas

        if self.fractal_type in ["Sierpinski 3D", "Árbol 3D", "Koch 3D"]:
            self.draw_fractal_3d()
        else:
            self.image = Image.new("RGB", (self.width, self.height), self.bg_color)
            self.draw = ImageDraw.Draw(self.image)
            if self.fractal_type == "Koch":
                self.draw_koch()
            elif self.fractal_type == "Sierpinski":
                self.draw_sierpinski()
            elif self.fractal_type == "Mandelbrot":
                self.draw_mandelbrot()
            elif self.fractal_type == "Julia":
                self.draw_julia()
            elif self.fractal_type == "Árbol":
                self.draw_tree()
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.image = self.photo

    def draw_fractal_3d(self):
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.view_init(elev=self.elev_3d, azim=self.azim_3d)
        ax.set_box_aspect([1,1,1])
        if self.fractal_type == "Sierpinski 3D":
            self.plot_sierpinski_3d(ax, depth=self.iterations)
        elif self.fractal_type == "Árbol 3D":
            self.plot_tree_3d(ax, depth=self.iterations)
        elif self.fractal_type == "Koch 3D":
            self.plot_koch_3d(ax, depth=self.iterations)
        # Aplica zoom y traslación 3D
        lim = 1.5 / self.zoom_3d
        ox, oy, oz = self.offset_3d
        ax.set_xlim(-lim+ox, lim+ox)
        ax.set_ylim(-lim+oy, lim+oy)
        ax.set_zlim(-lim+oz, lim+oz)
        ax.set_axis_off()
        self.mpl_canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().place(x=0, y=0, width=self.width, height=self.height)

    def plot_sierpinski_3d(self, ax, depth=3):
        def tetrahedron(vertices, color):
            faces = [
                [vertices[0], vertices[1], vertices[2]],
                [vertices[0], vertices[1], vertices[3]],
                [vertices[0], vertices[2], vertices[3]],
                [vertices[1], vertices[2], vertices[3]],
            ]
            for face in faces:
                tri = np.array(face)
                ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], color=color, alpha=0.7, linewidth=0.2, edgecolor='k')

        def sierpinski(v, d):
            if d == 0:
                tetrahedron(v, color='cyan')
            else:
                v0, v1, v2, v3 = v
                m01 = (v0 + v1) / 2
                m02 = (v0 + v2) / 2
                m03 = (v0 + v3) / 2
                m12 = (v1 + v2) / 2
                m13 = (v1 + v3) / 2
                m23 = (v2 + v3) / 2
                sierpinski([v0, m01, m02, m03], d-1)
                sierpinski([m01, v1, m12, m13], d-1)
                sierpinski([m02, m12, v2, m23], d-1)
                sierpinski([m03, m13, m23, v3], d-1)

        v0 = np.array([0, 0, 0])
        v1 = np.array([1, 0, 0])
        v2 = np.array([0.5, np.sqrt(3)/2, 0])
        v3 = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])
        sierpinski([v0, v1, v2, v3], depth)

    def plot_tree_3d(self, ax, depth=5):
        def branch(x, y, z, length, angle, tilt, d):
            if d == 0:
                return
            x2 = x + length * math.sin(angle) * math.cos(tilt)
            y2 = y + length * math.sin(angle) * math.sin(tilt)
            z2 = z + length * math.cos(angle)
            ax.plot([x, x2], [y, y2], [z, z2], color='green', linewidth=d)
            branch(x2, y2, z2, length*0.7, angle-0.3, tilt+0.3, d-1)
            branch(x2, y2, z2, length*0.7, angle-0.3, tilt-0.3, d-1)
        branch(0, 0, 0, 1, math.pi/2, 0, depth)

    def plot_koch_3d(self, ax, depth=3):
        # Dibuja un copo de nieve de Koch en 3D (en el plano XY, con altura Z=0)
        def koch3d(ax, p1, p2, level):
            if level == 0:
                xs = [p1[0], p2[0]]
                ys = [p1[1], p2[1]]
                zs = [p1[2], p2[2]]
                ax.plot(xs, ys, zs, color="#00e0ff", linewidth=2)
            else:
                dx = (p2[0] - p1[0]) / 3
                dy = (p2[1] - p1[1]) / 3
                dz = (p2[2] - p1[2]) / 3
                a = (p1[0] + dx, p1[1] + dy, p1[2] + dz)
                c = (p1[0] + 2*dx, p1[1] + 2*dy, p1[2] + 2*dz)
                # Calcular vértice del triángulo en 3D
                # Vector base
                vx, vy = c[0] - a[0], c[1] - a[1]
                # Perpendicular en XY
                angle = math.pi/3
                px = a[0] + vx * math.cos(angle) - vy * math.sin(angle)
                py = a[1] + vx * math.sin(angle) + vy * math.cos(angle)
                pz = a[2]  # plano Z=0
                b = (px, py, pz)
                koch3d(ax, p1, a, level-1)
                koch3d(ax, a, b, level-1)
                koch3d(ax, b, c, level-1)
                koch3d(ax, c, p2, level-1)

        # Triángulo equilátero en XY, Z=0
        size = 1.5
        h = size * math.sqrt(3) / 2
        p1 = (0, -h/3, 0)
        p2 = (-size/2, h*2/3, 0)
        p3 = (size/2, h*2/3, 0)
        koch3d(ax, p1, p2, depth)
        koch3d(ax, p2, p3, depth)
        koch3d(ax, p3, p1, depth)

    def draw_koch(self):
        # Dibuja un copo de nieve de Koch (Koch snowflake)
        # Triángulo equilátero centrado
        size = 400
        h = size * math.sqrt(3) / 2
        cx, cy = self.width // 2, self.height // 2
        p1 = (cx, cy - h / 3)
        p2 = (cx - size / 2, cy + h / 3)
        p3 = (cx + size / 2, cy + h / 3)
        # Dibuja los tres lados
        self.koch(p1, p2, self.iterations)
        self.koch(p2, p3, self.iterations)
        self.koch(p3, p1, self.iterations)

    def koch(self, p1, p2, level):
        if level == 0:
            p1t = self.transform_point(*p1)
            p2t = self.transform_point(*p2)
            self.draw.line([p1t, p2t], fill=self.color, width=2)
        else:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            a = (p1[0] + dx/3, p1[1] + dy/3)
            c = (p1[0] + 2*dx/3, p1[1] + 2*dy/3)
            angle = math.pi/3
            bx = a[0] + (c[0] - a[0]) * math.cos(angle) - (c[1] - a[1]) * math.sin(angle)
            by = a[1] + (c[0] - a[0]) * math.sin(angle) + (c[1] - a[1]) * math.cos(angle)
            b = (bx, by)
            self.koch(p1, a, level-1)
            self.koch(a, b, level-1)
            self.koch(b, c, level-1)
            self.koch(c, p2, level-1)

    def draw_sierpinski(self):
        p1 = (400, 100)
        p2 = (100, 500)
        p3 = (700, 500)
        self.sierpinski(p1, p2, p3, self.iterations)

    def sierpinski(self, p1, p2, p3, level):
        if level == 0:
            pts = [self.transform_point(*p1), self.transform_point(*p2), self.transform_point(*p3)]
            self.draw.polygon(pts, fill=self.color)
        else:
            # Puntos medios
            p12 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            p23 = ((p2[0]+p3[0])/2, (p2[1]+p3[1])/2)
            p31 = ((p3[0]+p1[0])/2, (p3[1]+p1[1])/2)
            self.sierpinski(p1, p12, p31, level-1)
            self.sierpinski(p12, p2, p23, level-1)
            self.sierpinski(p31, p23, p3, level-1)

    def draw_tree(self):
        start = (self.width // 2, self.height - 50)
        length = 150  # antes era 150 * self.zoom
        self.tree(start, length, -math.pi/2, self.iterations)


    def tree(self, start, length, angle, level):
        if level == 0:
            return
        end = (
            start[0] + length * math.cos(angle),
            start[1] + length * math.sin(angle)
        )
        p1t = self.transform_point(*start)
        p2t = self.transform_point(*end)
        self.draw.line([p1t, p2t], fill=self.color, width=max(1, level))
        new_length = length * 0.7
        self.tree(end, new_length, angle - math.pi/4, level-1)
        self.tree(end, new_length, angle + math.pi/4, level-1)

    def draw_mandelbrot(self):
        max_iter = self.iterations if hasattr(self, 'iterations') else 100
        arr = mandelbrot_escape(self.width, self.height, self.zoom, self.offset_x, self.offset_y, self.angle, max_iter)
        self.image = Image.fromarray(arr)


    def draw_julia(self):
        max_iter = self.iterations if hasattr(self, 'iterations') else 100
        # Parámetro C fijo (puedes hacerlo slider si quieres)
        arr = julia_escape(self.width, self.height, self.zoom, self.offset_x, self.offset_y, self.angle, -0.7, 0.27, max_iter)
        self.image = Image.fromarray(arr)
        
    def show_mandelbrot_3d(self):
        # Resolución reducida para vista rápida
        w, h = 200, 150
        max_iter = self.iterations if hasattr(self, 'iterations') else 100

        if self.fractal_type == "Julia":
            # Parámetro C fijo (puedes hacerlo slider si quieres)
            data = julia_escape(w, h, self.zoom, self.offset_x, self.offset_y, self.angle, -0.7, 0.27, max_iter)
            titulo = "Julia 3D"
        else:
            data = mandelbrot_escape(w, h, self.zoom, self.offset_x, self.offset_y, self.angle, max_iter)
            titulo = "Mandelbrot 3D"

        xs = np.linspace(-2.0/self.zoom, 2.0/self.zoom, w)
        ys = np.linspace(-2.0/self.zoom, 2.0/self.zoom, h)
        X, Y = np.meshgrid(xs, ys)
        Z = np.mean(data, axis=2)  # usar intensidad como altura

        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=False)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_zlabel('Altura')
        fig.colorbar(surf, ax=ax, shrink=0.5)

        top = tk.Toplevel(self.root)
        top.title(titulo)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)




    def transform_point(self, x, y):
        # Centrar el punto
        x -= self.width / 2
        y -= self.height / 2
        # Aplicar zoom
        x *= self.zoom
        y *= self.zoom
        # Aplicar rotación
        rad = math.radians(self.angle)
        x_rot = x * math.cos(rad) - y * math.sin(rad)
        y_rot = x * math.sin(rad) + y * math.cos(rad)
        # Aplicar traslación y volver a coordenadas de pantalla
        x_final = x_rot + self.width / 2 + self.offset_x
        y_final = y_rot + self.height / 2 + self.offset_y
        return (x_final, y_final)

    def change_fractal(self, event):
        self.fractal_type = self.fractal_var.get()
        self.draw_fractal()
        # Habilita solo si es Mandelbrot o Julia
        if self.fractal_type in ["Mandelbrot", "Julia"]:
            self.btn_3d.config(state="normal")
        else:
            self.btn_3d.config(state="disabled")
        self.master.focus_set()  # Quita el foco del combobox

    def update_iterations(self, value):
        self.iterations = int(value)
        self.draw_fractal()

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            if hasattr(self, 'image'):
                self.image.save(file_path)
            elif hasattr(self, 'mpl_canvas'):
                try:
                    self.mpl_canvas.figure.savefig(file_path, dpi=150, bbox_inches='tight')
                except Exception as e:
                    import tkinter.messagebox as messagebox
                    messagebox.showerror("Error", f"No se pudo guardar la figura 3D: {e}")


    def move_fractal(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self.draw_fractal()

    def rotate_fractal(self, da):
        self.angle = (self.angle + da) % 360
        self.draw_fractal()

    def zoom_in(self):
        self.zoom *= 1.1
        self.draw_fractal()

    def zoom_out(self):
        self.zoom /= 1.1
        self.draw_fractal()

    def move_3d(self, dx, dy, dz):
        self.offset_3d[0] += dx
        self.offset_3d[1] += dy
        self.offset_3d[2] += dz
        self.draw_fractal()

    def change_azim(self, da):
        self.azim_3d = (self.azim_3d + da) % 360
        self.draw_fractal()

    def change_elev(self, de):
        self.elev_3d = max(min(self.elev_3d + de, 90), -90)
        self.draw_fractal()

    def zoom_3d_in(self):
        self.zoom_3d *= 1.1
        self.draw_fractal()

    def zoom_3d_out(self):
        self.zoom_3d /= 1.1
        self.draw_fractal()

    def on_mousewheel(self, event):
        # Windows y Mac
        if hasattr(event, 'delta'):
            if event.delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        # Linux (event.num 4 = scroll up, 5 = scroll down)
        elif hasattr(event, 'num'):
            if event.num == 4:
                self.zoom_in()
            elif event.num == 5:
                self.zoom_out()

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalViewer(root)
    root.mainloop()
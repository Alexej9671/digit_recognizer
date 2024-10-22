import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageGrab, ImageTk
import numpy as np
import tensorflow as tf
import threading
import yaml
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Load configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load the constants from config.yaml
config = load_config('config.yml')

class DigitRecognizer:
    """
    Handwriting Predictor Application

    This application allows users to draw on a canvas, captures the drawing, and predicts the digit in real time using a pre-trained model.

    Attributes:
        model_path (str): Path to the pre-trained model.
    
    Methods:
        threaded_capture_canvas(): Capture the canvas in a separate thread.
        process_image(x, y, x1, y1): Process the captured image and predict the digit.
        update_histogram(prediction): Update the histogram on the UI.
        paint(event): Handle drawing on the canvas.
        clear_canvas(): Clear the drawing canvas.
        run(): Run the Tkinter main loop.
    """
    def __init__(self, model_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.stroke_count = 0  # Initialize stroke count
        
        # Create the main window
        self.root = tk.Tk()
        self._initialize_ui()  # Set up the user interface
        self._setup_bindings()  # Set up event bindings for user interaction
        
    def _initialize_ui(self):
        # Create the drawing frame
        frame_drawing = ttk.Frame(self.root)
        frame_drawing.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create the canvas for drawing
        self.canvas = tk.Canvas(frame_drawing, width=config['CANVAS_WIDTH'], height=config['CANVAS_HEIGHT'], background="black")
        self.canvas.pack()

        # Create the frame for the histogram
        frame_histogram = ttk.Frame(self.root)
        frame_histogram.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create the canvas for the histogram
        self.histogram_canvas = tk.Canvas(frame_histogram, width=config['HISTOGRAM_WIDTH'], height=config['HISTOGRAM_HEIGHT'], background="white")
        self.histogram_canvas.pack()

        # Add buttons for capturing and clearing the canvas
        button = tk.Button(self.root, text="Capture Canvas", command=self.threaded_capture_canvas)
        button.pack()
        
        clear_button = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas)
        clear_button.pack()
    
    def _setup_bindings(self):
        # Bind the paint method to mouse motion on the canvas
        self.canvas.bind("<B1-Motion>", self.paint)

    def threaded_capture_canvas(self):
        # Get the coordinates in the main thread
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + config['CANVAS_WIDTH']
        y1 = y + config['CANVAS_HEIGHT']
        # Start a background thread to process the image
        threading.Thread(target=self.process_image, args=(x, y, x1, y1)).start()

    def process_image(self, x, y, x1, y1):
        # Capture the content of the drawing canvas in the background thread
        image = ImageGrab.grab(bbox=(x, y, x1, y1))
        # Process the image
        image = image.resize((28, 28), Image.Resampling.NEAREST)
        gray_image = image.convert('L')  # Convert the image to grayscale
        gray_array = np.array(gray_image)  # Convert the image to a NumPy array
        
        # Due to anti-aliasing during resizing, the edges of the image may contain non-zero values.
        # To ensure clean edges, we manually set the edge values to zero.
        gray_array[0, :] = 0  # Set the first row to zero
        gray_array[-1, :] = 0  # Set the last row to zero
        gray_array[:, 0] = 0  # Set the first column to zero
        gray_array[:, -1] = 0  # Set the last column to zero
        
        # Reshape to (1, 28, 28, 1) to match the model's expected input shape
        arr = gray_array.reshape((1, 28, 28, 1))
        predictions = self.model.predict(arr)  # Get predictions from the model
        prediction = predictions[0]  # Extract the prediction probabilities

        # Now schedule the UI update on the main thread
        self.root.after(0, self.update_histogram, prediction)

    def update_histogram(self, prediction):
        # Create a Matplotlib figure and axis for the histogram
        fig, ax = plt.subplots(figsize=(2.8, 1.4), dpi=100)
        ax.bar(range(10), prediction)  # Plot the prediction probabilities as a bar chart
        ax.set_title('Prediction Probabilities')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probability')
        ax.set_xticks(range(10))
        plt.tight_layout()
        
        # Render the figure as an image on the Tkinter canvas
        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()
        tk_img = ImageTk.PhotoImage(Image.frombytes("RGB", canvas_agg.get_width_height(), canvas_agg.tostring_rgb()))
        # Update the existing histogram canvas and display the histogram
        self.histogram_canvas.delete("all")
        self.histogram_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.histogram_canvas.image = tk_img  # Keep a reference to avoid garbage collection
        
        # Close the figure to release memory
        plt.close(fig)

    def paint(self, event):
        # Get the current mouse position
        x1, y1 = (event.x - config['BRUSH_SIZE']), (event.y - config['BRUSH_SIZE'])
        x2, y2 = (event.x + config['BRUSH_SIZE']), (event.y + config['BRUSH_SIZE'])
        # Draw the oval (a small circle) at the current mouse position
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        
        # Increment stroke count
        self.stroke_count += 1
        
        # Call capture_canvas every set number of strokes
        if self.stroke_count >= config['STROKE_THRESHOLD']:
            self.threaded_capture_canvas()
            self.stroke_count = 0  # Reset stroke count

    def clear_canvas(self):
        # Clear the drawing canvas
        self.canvas.delete("all")

    def run(self):
        # Run the Tkinter main loop
        self.root.mainloop()

# Instantiate and run the application
if __name__ == "__main__":
    predictor = DigitRecognizer('model.keras')
    predictor.run()

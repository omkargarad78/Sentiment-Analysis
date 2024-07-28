import tkinter as tk
from tkinter import messagebox
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Function to predict sentiment
def predict_sentiment(user_input):
    # Tokenize the user input using the fine-tuned model's tokenizer
    tokens = tokenizer(user_input, return_tensors='pt')

    # Make a prediction
    outputs = model(**tokens)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"

    return sentiment

# Function to handle button click event
def analyze_sentiment():
    user_input = entry.get()
    if user_input.strip() == "":
        messagebox.showerror("Error", "Please enter a sentence.")
    else:
        sentiment_prediction = predict_sentiment(user_input)
        result_window = tk.Toplevel(root)
        result_window.title("Sentiment Analysis Result")

        # Center the result window on the screen
        result_window_width = 300
        result_window_height = 80
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - result_window_width) // 2
        y = (screen_height - result_window_height) // 2
        result_window.geometry(f"{result_window_width}x{result_window_height}+{x}+{y}")

        label = tk.Label(result_window, text=f"The sentiment of the sentence is {sentiment_prediction}.")
        label.pack(pady=15)
        root.after(3000, result_window.destroy)  # Close the result window after 3 seconds

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis")

# Set the geometry to center the window on the screen
window_width = 400
window_height = 150
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a label
label = tk.Label(root, text="Enter a sentence:")
label.pack(pady=10)

# Create an entry widget
entry = tk.Entry(root)
entry.pack()

# Create a button
button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
button.pack(pady=15)

# Run the Tkinter event loop
root.mainloop()

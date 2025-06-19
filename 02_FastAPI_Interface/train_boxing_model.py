import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from threading import Thread

class BoxingClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Boxing Pose Classifier Trainer")
        self.root.geometry("600x400")

        tk.Label(root, text="Boxing Pose Classifier Trainer", font=("Arial", 14)).pack(pady=10)

        tk.Button(root, text="Select JSON File", command=self.select_file).pack(pady=5)
        tk.Button(root, text="Select JSON Folder", command=self.select_folder).pack(pady=5)

        self.output_text = scrolledtext.ScrolledText(root, height=15, width=70, wrap=tk.WORD)
        self.output_text.pack(pady=10)

        self.train_button = tk.Button(root, text="Start Training", command=self.start_training, state=tk.DISABLED)
        self.train_button.pack(pady=5)

        self.data = []
        self.is_folder = False
        self.input_path = ""

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.input_path = file_path
            self.is_folder = False
            self.train_button.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"File selected: {file_path}\n")
            self.load_file_data()

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.input_path = folder_path
            self.is_folder = True
            self.train_button.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Folder selected: {folder_path}\n")
            self.load_folder_data()

    def load_file_data(self):
        try:
            with open(self.input_path, 'r') as f:
                self.data = json.load(f)
            self.output_text.insert(tk.END, f"Loaded {len(self.data)} samples from file.\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found.")
            self.data = []
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON file.")
            self.data = []

    def load_folder_data(self):
        try:
            json_files = [f for f in os.listdir(self.input_path) if f.endswith('.json')]
            if not json_files:
                messagebox.showerror("Error", "No JSON files found in folder.")
                self.data = []
                return
                
            self.data = []
            for json_file in json_files:
                with open(os.path.join(self.input_path, json_file), 'r') as f:
                    file_data = json.load(f)
                    self.data.extend(file_data)
            self.output_text.insert(tk.END, f"Loaded {len(self.data)} samples from {len(json_files)} files.\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "Folder not found.")
            self.data = []
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON file in folder.")
            self.data = []

    def start_training(self):
        self.train_button.config(state=tk.DISABLED)
        self.output_text.insert(tk.END, "Starting training with 5-fold cross-validation...\n")
        Thread(target=self.train_model).start()

    def train_model(self):
        features = []
        labels = []

        if not self.data:
            self.root.after(0, lambda: messagebox.showerror("Error", "No data loaded."))
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            return

        expected_num_landmarks = 6
        for sample in self.data:
            try:
                pose_name = sample['name']
                landmarks = sample['landmarks']
                
                if len(landmarks) != expected_num_landmarks:
                    self.output_text.insert(tk.END, f"Sample has wrong number of landmarks: {len(landmarks)} (expected {expected_num_landmarks})\n")
                    continue
                    
                landmark_coords = []
                for landmark in landmarks:
                    landmark_coords.extend([landmark['x'], landmark['y'], landmark['z']])
                    
                features.append(landmark_coords)
                labels.append(pose_name)
                
            except KeyError:
                self.output_text.insert(tk.END, "Missing key in sample.\n")
                continue

        if not features:
            self.root.after(0, lambda: messagebox.showerror("Error", "No valid data to train."))
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            return

        X = np.array(features)
        y = np.array(labels)

        self.root.after(0, lambda: self.output_text.insert(tk.END, f"Feature shape: {X.shape}\nNumber of labels: {len(y)}\n"))

        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_reports = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = MLPClassifier(alpha=1, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
            report = classification_report(y_test, y_pred, target_names=sorted(set(y)), output_dict=True)

            fold_reports.append((fold, accuracy, report))
            self.root.after(0, lambda: self.output_text.insert(tk.END, f"Fold {fold} - Accuracy: {accuracy:.2f}\n"))

        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        self.root.after(0, lambda: self.output_text.insert(tk.END, 
            f"\nMean Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}\n"))

        self.root.after(0, lambda: self.output_text.insert(tk.END, "\nClassification Report per Fold:\n"))
        for fold, accuracy, report in fold_reports:
            self.root.after(0, lambda: self.output_text.insert(tk.END, f"\nFold {fold} (Accuracy: {accuracy:.2f}):\n"))
            for pose, metrics in report.items():
                if isinstance(metrics, dict):
                    self.root.after(0, lambda: self.output_text.insert(tk.END, 
                        f"  {pose}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}\n"))

        final_model = MLPClassifier(alpha=1, max_iter=1000, random_state=42)
        final_model.fit(X, y)
        self.root.after(0, lambda: self.output_text.insert(tk.END, "\nTraining final model on complete dataset...\n"))

        joblib.dump(final_model, 'boxing_classifier.pkl')
        self.root.after(0, lambda: self.output_text.insert(tk.END, "Final model saved as 'boxing_classifier.pkl'\n"))

        self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: messagebox.showinfo("Success", "Training completed and model saved!"))

if __name__ == "__main__":
    root = tk.Tk()
    app = BoxingClassifierApp(root)
    root.mainloop()
from __future__ import annotations

import csv
import queue
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
from PIL import Image, ImageTk

from run_multimodal_video_inference import (
    MultiModalVideoInferencer,
    get_smoothed_prediction,
    open_video,
    read_frame_at,
    render_side_by_side,
)


class MultiModalGUI(tk.Tk):
    """A small local GUI for running paired body+face video inference."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Driver Multimodal Demo")
        self.geometry("1500x900")
        self.minsize(1280, 760)

        self.preview_image: ImageTk.PhotoImage | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_variables()
        self._build_layout()
        self.after(50, self._process_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_variables(self) -> None:
        root = Path(__file__).resolve().parent
        self.config_var = tk.StringVar(value=str(root / "train_multimodal_v14.yaml"))
        self.checkpoint_var = tk.StringVar(
            value=str(root / "runs" / "multimodal" / "train_multimodal_v14" / "weights" / "best.pt")
        )
        self.body_video_var = tk.StringVar(value=str(root / "runs" / "aligned" / "s1_body_aligned.mp4"))
        self.face_video_var = tk.StringVar(value=str(root / "runs" / "aligned" / "s1_face_aligned.mp4"))
        self.output_video_var = tk.StringVar(value=str(root / "runs" / "gui_outputs" / "demo_output_v14.mp4"))
        self.output_csv_var = tk.StringVar(value=str(root / "runs" / "gui_outputs" / "demo_output_v14.csv"))

        self.sample_interval_var = tk.IntVar(value=3)
        self.window_size_var = tk.IntVar(value=8)
        self.body_shift_var = tk.IntVar(value=0)
        self.face_shift_var = tk.IntVar(value=0)
        self.max_frames_var = tk.IntVar(value=0)
        self.save_video_var = tk.BooleanVar(value=True)
        self.save_csv_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame: - / -")
        self.distraction_var = tk.StringVar(value="Distraction: -")
        self.fatigue_var = tk.StringVar(value="Fatigue: -")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(self, padding=12)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_frame, text="Preview will appear here", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(self, padding=(0, 12, 12, 12))
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(1, weight=1)
        side.rowconfigure(9, weight=1)

        row = 0
        row = self._add_path_row(side, row, "Config", self.config_var, self._browse_config)
        row = self._add_path_row(side, row, "Checkpoint", self.checkpoint_var, self._browse_checkpoint)
        row = self._add_path_row(side, row, "Body Video", self.body_video_var, self._browse_body_video)
        row = self._add_path_row(side, row, "Face Video", self.face_video_var, self._browse_face_video)
        row = self._add_path_row(side, row, "Output Video", self.output_video_var, self._browse_output_video)
        row = self._add_path_row(side, row, "Output CSV", self.output_csv_var, self._browse_output_csv)

        params = ttk.LabelFrame(side, text="Inference Settings", padding=10)
        params.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 8))
        for col in range(4):
            params.columnconfigure(col, weight=1)

        ttk.Label(params, text="Sample Interval").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.sample_interval_var, width=8).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(params, text="Window Size").grid(row=0, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.window_size_var, width=8).grid(row=0, column=3, sticky="ew")

        ttk.Label(params, text="Body Shift").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(params, textvariable=self.body_shift_var, width=8).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        ttk.Label(params, text="Face Shift").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(params, textvariable=self.face_shift_var, width=8).grid(row=1, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(params, text="Max Frames (0=all)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(params, textvariable=self.max_frames_var, width=8).grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        ttk.Checkbutton(params, text="Save Video", variable=self.save_video_var).grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(params, text="Save CSV", variable=self.save_csv_var).grid(row=2, column=3, sticky="w", pady=(8, 0))

        row += 1

        buttons = ttk.Frame(side)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(buttons, text="Start", command=self.start_inference)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop_inference, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(buttons, text="Open Output Dir", command=self._open_output_dir_hint).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        row += 1

        status_box = ttk.LabelFrame(side, text="Current Status", padding=10)
        status_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_box, textvariable=self.frame_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(status_box, textvariable=self.distraction_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(status_box, textvariable=self.fatigue_var).grid(row=3, column=0, sticky="w", pady=(6, 0))

        row += 1

        log_box = ttk.LabelFrame(side, text="Logs", padding=10)
        log_box.grid(row=row, column=0, columnspan=3, sticky="nsew")
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)
        self.log_text = ScrolledText(log_box, wrap="word", height=18, font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _add_path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, browse_cmd) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(0, 6))
        ttk.Button(parent, text="Browse", command=browse_cmd).grid(row=row, column=2, sticky="ew", pady=(0, 6))
        return row + 1

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self.config_var.set(path)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.checkpoint_var.set(path)

    def _browse_body_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.body_video_var.set(path)

    def _browse_face_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.face_video_var.set(path)

    def _browse_output_video(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("All files", "*.*")])
        if path:
            self.output_video_var.set(path)

    def _browse_output_csv(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.output_csv_var.set(path)

    def _open_output_dir_hint(self) -> None:
        path = Path(self.output_video_var.get()).resolve().parent
        messagebox.showinfo("Output Directory", f"Output directory:\n{path}")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_inference(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Running", "Inference is already running.")
            return

        required = {
            "Config": self.config_var.get(),
            "Checkpoint": self.checkpoint_var.get(),
            "Body Video": self.body_video_var.get(),
            "Face Video": self.face_video_var.get(),
        }
        missing = [name for name, value in required.items() if not value or not Path(value).exists()]
        if missing:
            messagebox.showerror("Missing files", "These files are missing:\n" + "\n".join(missing))
            return

        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Initializing...")
        self._append_log("Starting inference...")

        self.worker_thread = threading.Thread(target=self._run_inference_worker, daemon=True)
        self.worker_thread.start()

    def stop_inference(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping...")
            self._append_log("Stop requested.")

    def _process_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "metrics":
                    data = payload
                    self.frame_var.set(f"Frame: body {data['body_frame_idx']} / face {data['face_frame_idx']}")
                    self.distraction_var.set(
                        f"Distraction: {data['distraction_pred']} ({data['distraction_conf']:.3f})"
                    )
                    self.fatigue_var.set(f"Fatigue: {data['fatigue_pred']} ({data['fatigue_conf']:.3f})")
                elif kind == "preview":
                    self._update_preview(payload)
                elif kind == "done":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set(str(payload))
                elif kind == "error":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set("Error")
                    messagebox.showerror("Inference Error", str(payload))
        except queue.Empty:
            pass
        self.after(50, self._process_ui_queue)

    def _update_preview(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        max_w = max(self.preview_label.winfo_width(), 800)
        max_h = max(self.preview_label.winfo_height(), 520)
        scale = min(max_w / image.width, max_h / image.height)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def _run_inference_worker(self) -> None:
        writer = None
        csv_file = None
        body_cap = None
        face_cap = None

        try:
            config_path = self.config_var.get()
            checkpoint_path = self.checkpoint_var.get()
            body_video = self.body_video_var.get()
            face_video = self.face_video_var.get()
            output_video = self.output_video_var.get().strip() if self.save_video_var.get() else ""
            output_csv = self.output_csv_var.get().strip() if self.save_csv_var.get() else ""

            inferencer = MultiModalVideoInferencer(config_path, checkpoint_path)
            body_cap = open_video(body_video)
            face_cap = open_video(face_video)

            fps = body_cap.get(cv2.CAP_PROP_FPS) or 25.0
            body_width = int(body_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            body_height = int(body_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            face_width = int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            face_height = int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_width = body_width + face_width
            out_height = max(body_height, face_height) + 140

            if output_video:
                output_video_path = Path(output_video)
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (out_width, out_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to create output video: {output_video_path}")

            csv_writer = None
            if output_csv:
                output_csv_path = Path(output_csv)
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_file = output_csv_path.open("w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "body_frame_idx",
                        "face_frame_idx",
                        "distraction_pred",
                        "distraction_conf",
                        "fatigue_pred",
                        "fatigue_conf",
                    ],
                )
                csv_writer.writeheader()

            sample_interval = max(1, int(self.sample_interval_var.get()))
            window_size = max(1, int(self.window_size_var.get()))
            body_shift = int(self.body_shift_var.get())
            face_shift = int(self.face_shift_var.get())
            max_frames = max(0, int(self.max_frames_var.get()))

        except Exception as exc:
            if body_cap is not None:
                body_cap.release()
            if face_cap is not None:
                face_cap.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
            self.ui_queue.put(("error", str(exc)))
            return

        distraction_window = deque(maxlen=window_size)
        fatigue_window = deque(maxlen=window_size)

        current_distraction_pred = "warming_up"
        current_distraction_conf = 0.0
        current_distraction_probs = np.zeros(len(inferencer.distraction_classes), dtype=np.float32)
        current_fatigue_pred = "warming_up"
        current_fatigue_conf = 0.0
        current_fatigue_probs = np.zeros(len(inferencer.fatigue_classes), dtype=np.float32)

        body_frame_idx = -1
        last_preview_ts = 0.0

        try:
            self.ui_queue.put(("status", "Running..."))
            while not self.stop_event.is_set():
                ok, body_frame = body_cap.read()
                if not ok:
                    break

                body_frame_idx += 1
                if max_frames and body_frame_idx >= max_frames:
                    break

                original_frame_idx = body_frame_idx + body_shift
                face_frame_idx = original_frame_idx - face_shift
                if face_frame_idx < 0:
                    continue

                face_frame = read_frame_at(face_cap, face_frame_idx)
                if face_frame is None:
                    break

                if body_frame_idx % sample_interval == 0:
                    preds = inferencer.predict(body_frame, face_frame)
                    distraction_window.append(preds["distraction_probs"])
                    fatigue_window.append(preds["fatigue_probs"])

                    current_distraction_pred, current_distraction_conf, current_distraction_probs = get_smoothed_prediction(
                        distraction_window, inferencer.distraction_classes
                    )
                    current_fatigue_pred, current_fatigue_conf, current_fatigue_probs = get_smoothed_prediction(
                        fatigue_window, inferencer.fatigue_classes
                    )

                canvas = render_side_by_side(
                    body_frame=body_frame,
                    face_frame=face_frame,
                    body_frame_idx=body_frame_idx,
                    face_frame_idx=face_frame_idx,
                    distraction_pred=current_distraction_pred,
                    distraction_conf=current_distraction_conf,
                    distraction_probs=current_distraction_probs,
                    distraction_classes=inferencer.distraction_classes,
                    fatigue_pred=current_fatigue_pred,
                    fatigue_conf=current_fatigue_conf,
                    fatigue_probs=current_fatigue_probs,
                    fatigue_classes=inferencer.fatigue_classes,
                )

                if writer is not None:
                    writer.write(canvas)
                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "body_frame_idx": body_frame_idx,
                            "face_frame_idx": face_frame_idx,
                            "distraction_pred": current_distraction_pred,
                            "distraction_conf": f"{current_distraction_conf:.6f}",
                            "fatigue_pred": current_fatigue_pred,
                            "fatigue_conf": f"{current_fatigue_conf:.6f}",
                        }
                    )

                self.ui_queue.put(
                    (
                        "metrics",
                        {
                            "body_frame_idx": body_frame_idx,
                            "face_frame_idx": face_frame_idx,
                            "distraction_pred": current_distraction_pred,
                            "distraction_conf": current_distraction_conf,
                            "fatigue_pred": current_fatigue_pred,
                            "fatigue_conf": current_fatigue_conf,
                        },
                    )
                )

                now = time.time()
                if now - last_preview_ts > 0.08:
                    self.ui_queue.put(("preview", canvas))
                    last_preview_ts = now

                if body_frame_idx % 120 == 0:
                    self.ui_queue.put(
                        (
                            "log",
                            f"body={body_frame_idx} face={face_frame_idx} "
                            f"distraction={current_distraction_pred}({current_distraction_conf:.3f}) "
                            f"fatigue={current_fatigue_pred}({current_fatigue_conf:.3f})",
                        )
                    )

            if self.stop_event.is_set():
                done_text = "Stopped by user."
            else:
                done_text = "Inference complete."
                if output_video:
                    done_text += f" Video: {output_video}"
                if output_csv:
                    done_text += f" CSV: {output_csv}"
            self.ui_queue.put(("log", done_text))
            self.ui_queue.put(("done", done_text))
        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))
        finally:
            body_cap.release()
            face_cap.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()

    def _on_close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.after(200, self.destroy)
        else:
            self.destroy()


def main() -> None:
    app = MultiModalGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

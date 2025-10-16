import os
import subprocess
import json
import sqlite3
from datetime import datetime

def extract_event_clips(input_video, event_times, output_dir, pre_event=2, post_event=2, db_path=None):

    os.makedirs(output_dir, exist_ok=True)
    conn = None
    cursor = None

    if db_path:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS clips (
                            id INTEGER PRIMARY KEY,
                            source_video TEXT,
                            clip_path TEXT,
                            event_type TEXT,
                            timestamp REAL,
                            importance_score REAL,
                            extracted_date TEXT
                          )''')
        conn.commit()

    for i, event in enumerate(event_times):
        start_time = max(0, event['timestamp'] - pre_event)
        duration = pre_event + post_event
        clip_filename = f"event_clip_{i:03d}.mp4"
        output_path = os.path.join(output_dir, clip_filename)

        command = [
            "ffmpeg",
            "-y",                     
            "-ss", str(start_time),
            "-i", input_video,
            "-t", str(duration),
            "-c:v", "libx264",       
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Saved {output_path}")

        if conn:
            cursor.execute('''INSERT INTO clips (source_video, clip_path, event_type, timestamp, importance_score, extracted_date)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (input_video, output_path, event['event_type'], event['timestamp'], event['importance_score'], datetime.now().isoformat()))
            conn.commit()

    if conn:
        conn.close()
        print(f"Metadata saved to {db_path}")
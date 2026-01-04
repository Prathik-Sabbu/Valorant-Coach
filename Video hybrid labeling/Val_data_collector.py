import json
import sqlite3
from Clip_shortener import extract_event_clips

class AutomatedValorantDataCollector:
    """
    Lightweight pipeline: just manages config, database,
    and stores clips/metadata into SQLite.
    """
    def __init__(self, config_path: str = r'C:\Users\sabbu\OneDrive\Documents\Git\Valorant-Coach\Video hybrid labeling\config.json'):
        self.config = self.load_config(config_path)
        self.setup_database()

    def load_config(self, config_path: str) -> dict:
        "Load configuration from file or defaults"
        default_config = {
            "storage": {
                "raw_clips_dir": "./raw_clips",
                "processed_data_dir": "./processed_data", 
                "database_path": "./valorant_data.db"
            }
        }

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            return default_config

    def setup_database(self):
        "Initialize the SQLite database with tables for clips"
        conn = sqlite3.connect(self.config['storage']['database_path'])
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY,
                source_video TEXT,
                clip_path TEXT,
                event_type TEXT,
                timestamp REAL,
                importance_score REAL,
                extracted_date TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def ingest_clips(self, input_video: str, event_times: list):
        """
        Cut clips from a video and insert metadata into database.
        Expects event_times = [{event_type, timestamp, importance_score}, ...]
        """
        output_dir = self.config['storage']['processed_data_dir']
        db_path = self.config['storage']['database_path']

        extract_event_clips(
            input_video=input_video,
            event_times=event_times,
            output_dir=output_dir,
            pre_event=2,
            post_event=2,
            db_path=db_path
        )

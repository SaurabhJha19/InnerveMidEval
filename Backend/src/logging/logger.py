import sqlite3
import json
import csv
from datetime import datetime


class Logger:
    def __init__(self):
        self.db = sqlite3.connect('detection_logs.db')
        self.cursor = self.db.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                is_deepfake BOOLEAN,
                confidence FLOAT,
                metadata TEXT
            )
        ''')
        self.db.commit()

    def log_incident(self, data):
        timestamp = datetime.now().isoformat()
        self.cursor.execute('''
            INSERT INTO detections (timestamp, is_deepfake, confidence, metadata)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, data['is_deepfake'], data['confidence'], json.dumps(data.get('metadata', {}))))
        self.db.commit()

    def export_to_json(self, filename='logs.json'):
        self.cursor.execute('SELECT * FROM detections')
        rows = self.cursor.fetchall()
        logs = [{
            'id': row[0],
            'timestamp': row[1],
            'is_deepfake': row[2],
            'confidence': row[3],
            'metadata': json.loads(row[4])
        } for row in rows]

        with open(filename, 'w') as f:
            json.dump(logs, f, indent=4)

    def export_to_csv(self, filename='logs.csv'):
        self.cursor.execute('SELECT * FROM detections')
        rows = self.cursor.fetchall()

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'timestamp', 'is_deepfake', 'confidence', 'metadata'])
            writer.writerows(rows)

    def __del__(self):
        self.db.close()
# db_handler.py
import sqlite3

class SQLiteHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self.schema = self.extract_schema()

    def extract_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        schema = {}
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table_name, in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema[table_name] = {
                "columns": [col[1] for col in columns],
                "types": [col[2] for col in columns],
                "primary_keys": [col[1] for col in columns if col[5] == 1]
            }
        conn.close()
        return schema

    def execute_query(self, query):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.close()
            return True, {"columns": columns, "data": results}, ""
        except Exception as e:
            return False, {}, str(e)

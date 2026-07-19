import os
import sqlite3
import json
import base64
import re

# Auto-detect database and media folder locations
db_path = None
possible_db_paths = [
    "resolvehub.db",
    "backend/resolvehub.db",
    "../resolvehub.db"
]

for path in possible_db_paths:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    for root, dirs, files in os.walk("."):
        if "resolvehub.db" in files:
            db_path = os.path.join(root, "resolvehub.db")
            break

if not db_path:
    db_path = "resolvehub.db"
    print("⚠️ Warning: resolvehub.db not found. A new database file will be initialized.")
else:
    print(f"Found database file: {os.path.abspath(db_path)}")

base_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."
media_dir = os.path.join(base_dir, "uploads", "offline_media")
os.makedirs(media_dir, exist_ok=True)

print("Starting DB migration...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Clean up base64 in offline_knowledge_entries
cursor.execute("SELECT id, response_json FROM offline_knowledge_entries;")
rows = cursor.fetchall()
print(f"Checking {len(rows)} offline knowledge entries...")

def save_b64_file(b64_str, prefix, ext):
    if not b64_str or len(b64_str.strip()) < 10:
        return None
    # Strip headers if present
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    try:
        data = base64.b64decode(b64_str)
        filename = f"{prefix}_{os.urandom(4).hex()}{ext}"
        filepath = os.path.join(media_dir, filename)
        with open(filepath, "wb") as f:
            f.write(data)
        # return relative path for URL
        return f"/uploads/offline_media/{filename}"
    except Exception as e:
        print(f"Error saving base64 file: {e}")
        return None

updated_count = 0
for row_id, resp_str in rows:
    if not resp_str:
        continue
    try:
        resp = json.loads(resp_str)
        changed = False
        
        # 1.1 image_base64
        if "image_base64" in resp and resp["image_base64"]:
            mime = resp.get("image_mime_type", "image/png")
            ext = ".png" if "png" in mime else ".jpg"
            path = save_b64_file(resp["image_base64"], f"entry_{row_id}_img", ext)
            if path:
                resp["image_path"] = path
                resp["image_base64"] = None
                changed = True
                
        # 1.2 video_base64
        if "video_base64" in resp and resp["video_base64"]:
            mime = resp.get("video_mime_type", "video/mp4")
            ext = ".mp4" if "mp4" in mime else ".webm"
            path = save_b64_file(resp["video_base64"], f"entry_{row_id}_vid", ext)
            if path:
                resp["video_path"] = path
                resp["video_base64"] = None
                changed = True
                
        # 1.3 video_step_images
        if "video_step_images" in resp and resp["video_step_images"]:
            for idx, step in enumerate(resp["video_step_images"]):
                if "imageBase64" in step and step["imageBase64"]:
                    mime = step.get("mimeType", "image/png")
                    ext = ".png" if "png" in mime else ".jpg"
                    path = save_b64_file(step["imageBase64"], f"entry_{row_id}_step_{idx}_img", ext)
                    if path:
                        step["localPath"] = path
                        step["imageBase64"] = ""
                        changed = True
                elif "image_base64" in step and step["image_base64"]:
                    mime = step.get("mime_type", "image/png")
                    ext = ".png" if "png" in mime else ".jpg"
                    path = save_b64_file(step["image_base64"], f"entry_{row_id}_step_{idx}_img", ext)
                    if path:
                        step["local_path"] = path
                        step["image_base64"] = ""
                        changed = True

        if changed:
            cursor.execute(
                "UPDATE offline_knowledge_entries SET response_json = ? WHERE id = ?;",
                (json.dumps(resp), row_id)
            )
            updated_count += 1
    except Exception as e:
        print(f"Error migrating row {row_id}: {e}")

print(f"Successfully migrated {updated_count} entries to disk files.")

# 2. Drop obsolete tables
obsolete_tables = [
    "community_field_cases",
    "community_case_solutions",
    "community_solution_feedback",
    "community_case_confirmations",
    "community_case_followups",
    "entreprendre_history",
    "chat_messages"
]

print("Dropping obsolete tables...")
for t in obsolete_tables:
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {t};")
        print(f"Dropped table {t}")
    except Exception as e:
        print(f"Error dropping table {t}: {e}")

# Save changes
conn.commit()

# 3. Vacuum database to reclaim space
print("Vacuuming SQLite database...")
try:
    cursor.execute("VACUUM;")
    print("Database vacuumed successfully.")
except Exception as e:
    print(f"Error vacuuming database: {e}")

conn.close()

# Verify new file size
new_size = os.path.getsize(db_path)
print(f"Migration completed. New DB file size: {new_size} bytes ({new_size / 1024 / 1024:.2f} MB)")

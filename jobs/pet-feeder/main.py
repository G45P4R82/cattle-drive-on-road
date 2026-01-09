import time
import json
import os
import requests
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from rich.console import Console

console = Console()

from os import environ

# ================= Configuration =================
RTSP_URL = environ.get("RTSP_URL")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
STATE_FILE = os.path.join(BASE_DIR, "monitor_state.json")

# Thingsboard
TB_URL = environ.get("TB_URL")

# Discord
DISCORD_WEBHOOK = environ.get("DISCORD_WEBHOOK")

# Parameters
FRAMES_TO_PROCESS = 20
CONFIDENCE_THRESHOLD = 0.50 # Increased to avoid ghost detections

# Classes
CLASS_VAZIO = 0
CLASS_RACAO = 1
# =================================================

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def send_thingsboard(telemetry):
    try:
        response = requests.post(TB_URL, json=telemetry)
        if response.status_code == 200:
            console.log(f"[TB] Data sent successfully: {telemetry}")
        else:
            console.log(f"[TB] Failed to send data: {response.status_code} {response.text}")
    except Exception as e:
        console.log(f"[TB] Error sending data: {e}")

def send_discord(message, embed=None, image_path=None):
    # Construct the JSON payload
    payload = {"content": message}
    if embed:
        payload["embeds"] = [embed]
    
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                # To send JSON + File, Discord requires 'payload_json' field
                files = {
                    "file": (os.path.basename(image_path), f, "image/jpeg")
                }
                # We must accept that data expects 'payload_json' as a string
                data = {"payload_json": json.dumps(payload)}
                
                requests.post(DISCORD_WEBHOOK, data=data, files=files)
        else:
            # Just JSON
            requests.post(DISCORD_WEBHOOK, json=payload)
            
        console.log(f"[Discord] Alert sent.")
    except Exception as e:
        console.log(f"[Discord] Error sending alert: {e}")

def run_inference_cycle():
    now = datetime.now()
    current_hour = now.hour
    
    # Operating Hours: 06:00 to 21:00 (i.e., runs until 20:59)
    # User said "not run between 21 and 06"
    if current_hour < 6 or current_hour >= 21:
        console.log(f"Outside operating hours (06h-21h). Skipping.")
        return

    console.log(f"Starting Inference Cycle...")
    
    if not os.path.exists(MODEL_PATH):
        console.log(f"Error: Model not found at {MODEL_PATH}")
        send_discord(f"🚨 **CRITICAL ERROR**: Model not found at `{MODEL_PATH}`. Monitoring stopped.")
        return

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        console.log(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        console.log("Error: Could not open RTSP stream.")
        send_discord("⚠️ **WARNING**: Could not connect to RTSP stream.")
        return

    # Data collection for this cycle
    detections = [] # list of (class, confidence, id) if tracking enabled
    last_annotated_frame = None
    
    frames_processed = 0
    while frames_processed < FRAMES_TO_PROCESS:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run Tracking
        try:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Save visual for alert
            # We overwrite this each frame, so we get the *last* frame of the batch
            last_annotated_frame = results[0].plot()
            
            # Extract boxes
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                
                for i in range(len(ids)):
                    detections.append({
                        "id": int(ids[i]),
                        "cls": int(clss[i]),
                        "conf": float(confs[i])
                    })
            
            frames_processed += 1
            # Small delay to mimic real-time sampling spread if needed, but for speed we just run
        except Exception as e:
            console.log(f"Inference Loop Error: {e}")
            break
            
    cap.release()
    
    if frames_processed == 0:
        console.log("No frames processed.")
        return

    # --- Analysis ---
    # Aggregate by ID (Voting mechanism)
    objects = {} # id -> list of class votes
    
    for d in detections:
        oid = d['id']
        if oid not in objects:
            objects[oid] = {"votes": [], "confs": []}
        objects[oid]["votes"].append(d["cls"])
        objects[oid]["confs"].append(d["conf"])
        
    final_objects = []
    total_conf = 0
    count_conf = 0
    
    for oid, data in objects.items():
        # Majority vote for class
        class_id = max(set(data["votes"]), key=data["votes"].count)
        avg_conf = sum(data["confs"]) / len(data["confs"])
        
        final_objects.append({"id": oid, "cls": class_id, "conf": avg_conf})
        
        total_conf += sum(data["confs"])
        count_conf += len(data["confs"])

    # Metrics
    total_potes = len(final_objects)
    
    # HARD FILTER: If > 5 pots, try to drop lowest confidence ones or just warn
    # But increasing threshold is better. 
    # Let's sort by ID just for display consistency
    final_objects.sort(key=lambda x: x['id'])
    
    empty_potes = [o for o in final_objects if o['cls'] == CLASS_VAZIO]
    num_empty = len(empty_potes)
    pct_empty = (num_empty / total_potes * 100) if total_potes > 0 else 0
    avg_confidence = (total_conf / count_conf) if count_conf > 0 else 0
    
    empty_ids = [str(o['id']) for o in empty_potes]
    
    console.log(f"Result: {num_empty}/{total_potes} Empty ({pct_empty:.1f}%) | Health: {avg_confidence:.2f}")

    # --- Persistence & Logic ---
    state = load_state()
    # Simple logic: If we see ANY empty pot, we update the timestamp of "last seen empty"
    # Ideally we track per ID, but IDs drift on restart.
    # Let's track "Session State"
    
    timestamp = datetime.now().isoformat()
    
    # Telemetry Payload
    telemetry = {
        "potes_total": total_potes,
        "potes_empty": num_empty,
        "pct_empty": round(pct_empty, 1),
        "model_health": round(avg_confidence, 3),
        "empty_ids": ",".join(empty_ids)
    }
    
    send_thingsboard(telemetry)
    
    # --- Alert Logic & Suppression ---
    # Load previous state to check for changes
    last_num_empty = state.get("last_num_empty", -1)
    
    should_alert = False
    alerts = []
    
    # Defaults
    color = 0x00ff00 # Green
    
    # Threshold Rules
    if num_empty == 4:
        color = 0xffa500 # Yellow/Orange
        if num_empty != last_num_empty:
            alerts.append(f"⚠️ **Alerta (Amarelo)**: 4 Potes Vazios.")
            should_alert = True
            
    elif num_empty >= 5:
        color = 0xff0000 # Red
        if num_empty != last_num_empty:
            alerts.append(f"🚨 **Alerta (Vermelho)**: {num_empty} Potes Vazios!")
            should_alert = True
            
    else:
        # Green / Normal / Recovery
        if last_num_empty >= 4:
             alerts.append("✅ **Recuperação**: Situação normalizada (Menos de 4 vazios).")
             should_alert = True
    
    # Health Check (Independent)
    if avg_confidence < 0.75:
        color = 0xff0000 
        alerts.append(f"⚠️ **Alerta de Saúde**: Confiança baixa ({avg_confidence:.2f}).")
        should_alert = True

    # Construct Embed
    embed = {
        "title": "📊 Relatório de Monitoramento",
        "color": color,
        "fields": [
            {"name": "Status", "value": f"**{num_empty}** de **{total_potes}** potes vazios ({pct_empty:.0f}%)", "inline": True},
            {"name": "IDs Vazios", "value": ", ".join(empty_ids) if empty_ids else "Nenhum", "inline": True},
            {"name": "Saúde do Modelo", "value": f"{avg_confidence:.2f}", "inline": True},
            {"name": "Timestamp", "value": datetime.now().strftime("%H:%M:%S"), "inline": False}
        ]
    }

    # Send only if priority alert or significant change
    if should_alert:
        msg_content = "\n".join(alerts)
        
        # Save image if we have one
        img_path = None
        if last_annotated_frame is not None:
            img_path = "alert_snapshot.jpg"
            cv2.imwrite(img_path, last_annotated_frame)
        
        send_discord(msg_content, embed, image_path=img_path)
        send_discord(msg_content, embed, image_path=img_path)
    else:
        # Console.log adds timestamp automatically
        console.log("[Discord] Skipping alert (No change / No condition met)")

    # Update State
    state["last_num_empty"] = num_empty
    save_state(state)


def main():
    console.log("Starting Inference Job...")
    run_inference_cycle()
    console.log("Job Finished.")

if __name__ == "__main__":
    main()

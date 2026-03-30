#!/usr/bin/env python3
import json
import os
from datetime import datetime

def save_current_topic(user_id, topic_id, topic_name, mastery_score):
    kg_path = f"/root/data/AI/flyllm/.claude/skills/interview-master/memory/{user_id}/knowledge_graph.json"
    if os.path.exists(kg_path):
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
        
        kg['concepts'][topic_id] = {
            "concept_id": topic_id,
            "name": topic_name,
            "mastery_score": mastery_score,
            "last_interaction": datetime.now().strftime('%Y-%m-%d'),
            "status": "completed"
        }
        
        completed = len([c for c in kg['concepts'].values() if c.get('status') == 'completed'])
        kg['progress_tracking']['completed'] = completed
        kg['progress_tracking']['completion_rate'] = completed / 60
        
        with open(kg_path, 'w', encoding='utf-8') as f:
            json.dump(kg, f, ensure_ascii=False, indent=2)
        print(f"✅ 知识图谱已更新: {topic_name}")

if __name__ == '__main__':
    save_current_topic('Rain', 'tokenizer', 'Tokenizer核心机制', 0.75)

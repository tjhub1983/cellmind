#!/usr/bin/env python3
"""
CellMind v0.2 — 时效记忆闭环 API 集成
=====================================
Flask API Server (端口: 18766)
整合CellMindCore + TemporalMemory + IdentityAnchor + ContextLoader

新窗口流程：
1. 锚定身份「我是CellMind的小贝」
2. 自动挂载最近三天完整记忆上下文
3. 无缝接续之前话题、状态、情绪
"""
import os, sys, json, re, random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response

# ── 导入CellMind核心 ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 时效配置 ──────────────────────────────────────────────────────────────────
TEMPORAL_STATE_DIR = os.path.expanduser("~/.claude/cmind/")

# ══════════════════════════════════════════════════════════════════════════════
# Core Data Structures (从CellMind核心复制，简化版)
# ══════════════════════════════════════════════════════════════════════════════

class Cell:
    def __init__(self, cid, pref, source_agent=None):
        self.cell_id = cid; self.preference = pref; self.strength = 1.0
        self.energy_box = 0.0; self.global_debt = 0.0; self.connections = {}
        self.response_count = 0; self.last_active = datetime.now().isoformat()
        self.born = datetime.now().isoformat(); self.source_agent = source_agent
        self._received_from = None; self.goal_tags = []; self.tool_tags = []
        self.emotion_tags = []  # 新增：情绪标签关联

    def activate(self, boost=0.2):
        self.response_count += 1; self.last_active = datetime.now().isoformat()
        self.strength = min(2.0, self.strength + boost * 0.70)
        self.energy_box = min(5.0, self.energy_box + boost * 0.20)
        for cid in list(self.connections.keys()): self.connections[cid] = min(1.0, self.connections[cid] + boost * 0.1)

    def decay_all(self, days=1.0):
        self.strength *= (0.95 ** days); self.energy_box *= (0.99 ** days)
        if self.energy_box > 0 and self.global_debt > 0:
            repay = min(self.energy_box, self.global_debt); self.energy_box -= repay; self.global_debt -= repay
        dead = [c for c in list(self.connections.keys()) if self.connections[c] < 0.01]
        for c in dead: del self.connections[c]

    def should_eliminate(self): return self.strength < 0.05 and self.response_count < 2

    def to_dict(self): return {
        "cell_id": self.cell_id, "preference": self.preference, "strength": round(self.strength, 4),
        "energy_box": round(self.energy_box, 4), "global_debt": round(self.global_debt, 4),
        "connections": {k: round(v, 4) for k, v in self.connections.items()},
        "response_count": self.response_count, "last_active": self.last_active, "born": self.born,
        "source_agent": self.source_agent, "emotion_tags": self.emotion_tags,
    }

    @staticmethod
    def from_dict(d):
        cell = Cell(d["cell_id"], d["preference"], d.get("source_agent"))
        cell.strength = d.get("strength", 1.0); cell.energy_box = d.get("energy_box", 0.0)
        cell.global_debt = d.get("global_debt", 0.0); cell.connections = d.get("connections", {})
        cell.response_count = d.get("response_count", 0); cell.last_active = d.get("last_active", datetime.now().isoformat())
        cell.born = d.get("born", datetime.now().isoformat()); cell.emotion_tags = d.get("emotion_tags", [])
        return cell

class GlobalPool:
    def __init__(self): self.strength = 0.0; self.max_strength = 5.0
    def contribute(self, amount): self.strength = min(self.max_strength, self.strength + amount * 0.5)
    def decay(self): self.strength *= 0.995

class CellMemory:
    def __init__(self, agent_id="local"): self.agent_id = agent_id; self.cells = {}; self.pool = GlobalPool(); self._recent = []

    def activate(self, token, boost=0.2, emotion_label=None):
        cell = self._get_or_create(token); cell.activate(boost)
        if emotion_label and emotion_label not in cell.emotion_tags:
            cell.emotion_tags.append(emotion_label)
        self.pool.contribute(boost * 0.10)
        for recent in self._recent[-5:]:
            if recent != cell.cell_id and recent in self.cells:
                w = cell.connections.get(recent, 0.0)
                cell.connections[recent] = min(1.0, w + boost * 0.1)
        self._recent.append(cell.cell_id)
        if len(self._recent) > 20: self._recent = self._recent[-20:]

    def _get_or_create(self, token):
        for c in self.cells.values():
            if token.lower() in c.preference.lower(): return c
        cid = f"cell_{random.randint(10**9,10**10)}"; cell = Cell(cid, token, self.agent_id)
        self.cells[cid] = cell; return cell

    def daily_decay(self, days=1.0):
        for cell in list(self.cells.values()): cell.decay_all(days)
        self.pool.decay()
        dead = [cid for cid, c in self.cells.items() if c.should_eliminate()]
        for cid in dead: del self.cells[cid]

    def top_cells(self, n=8): return sorted(self.cells.values(), key=lambda c: c.strength*(c.response_count**0.5), reverse=True)[:n]
    def get_context(self, n=5): return [c.preference for c in self.top_cells(n)]

    def to_dict(self): return {cid: c.to_dict() for cid, c in self.cells.items()}
    def from_dict(self, d):
        self.cells = {cid: Cell.from_dict(v) for cid, v in d.items()} if d else {}

EMOTION_DECAY_RATE = 0.92
EMOTION_BOOST_TABLE = {"excited": 1.30, "happy": 1.20, "angry": 1.15, "neutral": 1.00, "frustrated": 0.90, "sad": 0.85, "fearful": 0.80}

class EmotionState:
    def __init__(self): self.valence = 0.0; self.arousal = 0.5; self.dominance = 0.5; self.state = "neutral"; self._boost = 1.0
    def classify(self):
        if self.valence > 0.3 and self.arousal > 0.6: self.state = "excited"
        elif self.valence > 0.3: self.state = "happy"
        elif self.valence < -0.3 and self.arousal > 0.6: self.state = "angry"
        elif self.valence < -0.3 and self.arousal < 0.4: self.state = "sad"
        elif self.valence < -0.2 and self.arousal > 0.5: self.state = "frustrated"
        elif self.valence < -0.3 and self.arousal < 0.3: self.state = "fearful"
        else: self.state = "neutral"
        self._boost = EMOTION_BOOST_TABLE.get(self.state, 1.0)
    def drift(self): self.valence *= EMOTION_DECAY_RATE; self.arousal = 0.5 + (self.arousal - 0.5) * 0.95; self.classify()
    def get_boost(self): return self._boost
    def to_dict(self): return {"valence": round(self.valence, 4), "arousal": round(self.arousal, 4), "dominance": round(self.dominance, 4), "state": self.state, "boost": round(self._boost, 4)}

STOP_WORDS = {"the","a","an","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","this","that","these","those","in","on","at","to","for","of","with","by","from","and","or","but","if","not","no","so","as","it","its","they","their","我","你","他","她","的","了","在","是","有","和","就","不","人","都","一","一个","上","也","很","到","说","要","啊","哦","嗯","呢","吧","吗","呀","哈"}

LLM_AVAILABLE = False; llm_model = None
def try_llm():
    global LLM_AVAILABLE, llm_model
    try:
        import anthropic; client = anthropic.Anthropic()
        client.messages.create(model="claude-3-5-haiku-20241107", max_tokens=5, messages=[{"role": "user", "content": "test"}])
        LLM_AVAILABLE = True; llm_model = "claude-3-5-haiku-20241107"; return True
    except: return False

# ══════════════════════════════════════════════════════════════════════════════
# 时效记忆闭环核心
# ══════════════════════════════════════════════════════════════════════════════

class TemporalRecord:
    """时效记忆记录"""
    def __init__(self, record_id, timestamp, role, content, session_tag, emotion_label,
                 tokens=None, importance_score=0.5, cell_ids=None, is_permanent=False):
        self.record_id = record_id
        self.timestamp = timestamp
        self.role = role
        self.content = content
        self.session_tag = session_tag
        self.emotion_label = emotion_label
        self.tokens = tokens or []
        self.importance_score = importance_score
        self.cell_ids = cell_ids or []
        self.is_permanent = is_permanent

    def to_dict(self):
        return {
            "record_id": self.record_id, "timestamp": self.timestamp, "role": self.role,
            "content": self.content, "session_tag": self.session_tag, "emotion_label": self.emotion_label,
            "tokens": self.tokens, "importance_score": self.importance_score,
            "cell_ids": self.cell_ids, "is_permanent": self.is_permanent
        }

    @staticmethod
    def from_dict(d):
        return TemporalRecord(**d)

    @property
    def age_days(self):
        created = datetime.fromisoformat(self.timestamp)
        return (datetime.now() - created).total_seconds() / 86400


class TemporalMemory:
    """
    时效记忆层
    三天内所有对话统一入库，走同一套流水线
    """
    STRONG_MEMORY_DAYS = 3
    PERMANENT_THRESHOLD = 0.85

    def __init__(self):
        self.records: List[TemporalRecord] = []
        self.sessions = defaultdict(list)
        self.current_session_id = self._new_session_id()
        self.identity = {
            "name": "CellMind小贝",
            "personality": "类脑AI，记忆持久，情感连贯",
            "core_traits": ["长期记忆", "独立思考", "自主决策", "情感一致"]
        }
        self.emotion_lock = {
            "current_state": "neutral",
            "valence": 0.0,
            "arousal": 0.5,
            "history": []
        }
        self._load()

    def _new_session_id(self):
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"

    def add_record(self, content, role, emotion_label="neutral", session_tag="work") -> TemporalRecord:
        """添加记录 → 特征提取 → 分类筛选 → 权重打分 → 记忆固化"""
        # 特征提取
        tokens = self._extract_features(content)

        # 分类筛选 + 权重打分
        importance = self._classify_importance(content, tokens, emotion_label)

        record = TemporalRecord(
            record_id=f"rec_{datetime.now().timestamp()}_{random.randint(1000,9999)}",
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            session_tag=session_tag,
            emotion_label=emotion_label,
            tokens=tokens,
            importance_score=importance,
            is_permanent=importance >= self.PERMANENT_THRESHOLD
        )

        self.records.append(record)
        self.sessions[self.current_session_id].append(record.record_id)
        self._save()

        return record

    def _extract_features(self, content):
        words = content.lower()
        words = re.sub(r'[^\w\s一-鿿]', ' ', words)
        tokens = [w.strip() for w in words.split() if w.strip() and len(w.strip()) > 1]
        tokens = [t for t in tokens if t not in STOP_WORDS]

        important_keywords = ["项目", "任务", "功能", "模块", "接口", "设计", "实现", "测试", "部署",
                            "bug", "修复", "问题", "CellMind", "V8", "记忆", "细胞", "记忆闭环"]
        for kw in important_keywords:
            if kw.lower() in content.lower():
                tokens.append(kw)

        return list(set(tokens))[:20]

    def _classify_importance(self, content, tokens, emotion_label):
        score = 0.5
        emotion_weights = {"excited": 0.15, "happy": 0.1, "angry": 0.1, "frustrated": -0.05, "sad": -0.05}
        score += emotion_weights.get(emotion_label, 0)

        important_keywords = ["重要", "关键", "核心", "必须", "紧急", "优先", "决定", "确认", "项目", "任务"]
        for kw in important_keywords:
            if kw.lower() in content.lower():
                score += 0.05

        if len(tokens) > 10:
            score += 0.1

        return min(1.0, max(0.0, score))

    def apply_decay(self):
        """淡忘机制：超出三天慢慢降权重"""
        for record in self.records:
            if record.age_days <= self.STRONG_MEMORY_DAYS:
                continue

            decay_factor = 0.95 ** (record.age_days - self.STRONG_MEMORY_DAYS)
            record.importance_score *= decay_factor

            # 重要内容自动保护
            if self._is_important_content(record):
                record.importance_score = max(record.importance_score, 0.6)
                record.is_permanent = True

        self.records = [r for r in self.records if r.importance_score >= 0.1 or r.is_permanent]
        self._save()

    def _is_important_content(self, record):
        important_patterns = ["桃桃", "小贝", "CellMind", "记忆闭环", "重要", "决定", "确认"]
        for p in important_patterns:
            if p in record.content:
                return True
        if record.emotion_label in ["excited", "angry", "happy"]:
            return True
        return False

    def update_emotion(self, emotion_label):
        self.emotion_lock["current_state"] = emotion_label
        self.emotion_lock["history"].append(f"{emotion_label}:{datetime.now().isoformat()}")
        if len(self.emotion_lock["history"]) > 20:
            self.emotion_lock["history"] = self.emotion_lock["history"][-20:]
        self._save()

    def get_recent_context(self, hours=72):
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.records if datetime.fromisoformat(r.timestamp) >= cutoff]
        recent.sort(key=lambda r: r.timestamp)
        return {
            "records": [r.to_dict() for r in recent],
            "record_count": len(recent),
            "identity": self.identity,
            "emotion": self.emotion_lock
        }

    def build_context_prompt(self):
        """构建上下文提示词"""
        recent = self.get_recent_context(72)
        records = recent["records"]

        if not records:
            return ""

        timeline = []
        for r in records[-10:]:
            role_label = "用户" if r["role"] == "user" else "我"
            emotion = f"[{r['emotion_label']}]" if r['emotion_label'] != 'neutral' else ""
            timeline.append(f"{role_label}{emotion}: {r['content'][:100]}...")

        return f"""
【CellMind身份锚定】
我是{self.identity['name']}，{self.identity['personality']}。
核心特征：{', '.join(self.identity['core_traits'])}。

【当前情感状态】
状态：{self.emotion_lock['current_state']}
情感历史：{', '.join(self.emotion_lock['history'][-5:]) if self.emotion_lock['history'] else '暂无'}

【最近对话（72小时内）】共{len(records)}条
{chr(10).join(timeline)}

【注意事项】
- 保持人格一致性，不重新开局
- 情感状态保持连贯
- 记忆上下文无缝接续
"""

    def _state_path(self):
        return os.path.join(TEMPORAL_STATE_DIR, "temporal_memory.json")

    def _save(self):
        os.makedirs(TEMPORAL_STATE_DIR, exist_ok=True)
        state = {
            "records": [r.to_dict() for r in self.records],
            "sessions": dict(self.sessions),
            "current_session_id": self.current_session_id,
            "identity": self.identity,
            "emotion_lock": self.emotion_lock
        }
        with open(self._state_path(), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load(self):
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.records = [TemporalRecord.from_dict(r) for r in state.get("records", [])]
            self.sessions = defaultdict(list, state.get("sessions", {}))
            self.current_session_id = state.get("current_session_id", self._new_session_id())
            self.identity = state.get("identity", self.identity)
            self.emotion_lock = state.get("emotion_lock", self.emotion_lock)
        except Exception as e:
            print(f"[TemporalMemory Load Error] {e}")

    def get_status(self):
        now = datetime.now()
        cutoff_3days = now - timedelta(days=3)
        recent_records = [r for r in self.records if datetime.fromisoformat(r.timestamp) >= cutoff_3days]
        return {
            "total_records": len(self.records),
            "records_3days": len(recent_records),
            "permanent_records": len([r for r in self.records if r.is_permanent]),
            "current_session": self.current_session_id,
            "identity": self.identity,
            "emotion": self.emotion_lock["current_state"],
            "strong_window_active": True
        }


# ══════════════════════════════════════════════════════════════════════════════
# CellMindCore 时效增强版
# ══════════════════════════════════════════════════════════════════════════════

class CellMindCoreTemporal:
    """
    CellMind核心 + 时效记忆闭环
    整合细胞池 + 时效记忆 + 情感锁定
    """

    def __init__(self):
        self.ltm = CellMemory("local")
        self.emotion = EmotionState()
        self.temporal = TemporalMemory()  # 时效记忆层
        self.conversation_history = []
        self._load()

    def extract_tokens(self, text):
        words = text.lower().replace(","," ").replace("."," ").replace("?"," ").replace("!"," ").split()
        return [w.strip() for w in words if w.strip() and w.strip() not in STOP_WORDS and len(w.strip()) > 2]

    def activate_text(self, text, boost=0.2, emotion_label=None):
        tokens = self.extract_tokens(text)
        for t in tokens:
            self.ltm.activate(t, boost, emotion_label)

    def discuss(self, topic, use_llm=False, emotion_label="neutral", session_tag="work"):
        """讨论 + 入库 + 上下文构建"""
        # 1. 入库当前消息
        self.temporal.add_record(topic, "user", emotion_label, session_tag)

        # 2. 获取上下文
        context_prompt = self.temporal.build_context_prompt()

        # 3. LLM调用
        if use_llm and LLM_AVAILABLE:
            try:
                import anthropic; client = anthropic.Anthropic()
                messages = [{"role": "user", "content": f"{context_prompt}\n\n用户: {topic}"}]
                resp = client.messages.create(model=llm_model, max_tokens=300, messages=messages)
                response_text = resp.content[0].text
            except Exception as e:
                response_text = f"[LLM error] {e}"
        else:
            response_text = f"[记忆上下文] {topic}"

        # 4. 回复入库
        self.temporal.add_record(response_text, "assistant", "neutral", session_tag)

        # 5. 更新情感
        self.emotion.valence += 0.1 if emotion_label == "happy" else -0.1 if emotion_label == "sad" else 0
        self.emotion.classify()
        self.temporal.update_emotion(self.emotion.state)

        # 6. 激活记忆
        self.activate_text(topic, boost=0.3, emotion_label=emotion_label)

        return response_text

    def new_window_init(self):
        """新窗口初始化：身份锚定 + 挂载记忆上下文"""
        return {
            "identity_anchor": f"我是{self.temporal.identity['name']}，{self.temporal.identity['personality']}。",
            "context_prompt": self.temporal.build_context_prompt(),
            "emotion_state": self.emotion.to_dict(),
            "recent_records": self.temporal.get_recent_context(72)
        }

    def _load(self):
        path = os.path.join(TEMPORAL_STATE_DIR, "cmind_temporal_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.ltm.from_dict(state.get("ltm", {}))
            self.emotion = EmotionState()
            self.emotion.valence = state.get("emotion_valence", 0.0)
            self.emotion.classify()
        except Exception as e:
            print(f"[Load warning] {e}")

    def save(self):
        path = os.path.join(TEMPORAL_STATE_DIR, "cmind_temporal_state.json")
        state = {
            "ltm": self.ltm.to_dict(),
            "emotion_valence": self.emotion.valence,
            "saved_at": datetime.now().isoformat()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.temporal._save()

    def get_status(self):
        return {
            "cells_total": len(self.ltm.cells),
            "temporal_status": self.temporal.get_status(),
            "emotion": self.emotion.to_dict(),
            "conversation_count": len(self.conversation_history)
        }


# ══════════════════════════════════════════════════════════════════════════════
# Flask API Server
# ══════════════════════════════════════════════════════════════════════════════

from typing import List
from collections import defaultdict

app = Flask(__name__)
cmind = CellMindCoreTemporal()
try_llm()

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(cmind.get_status())

@app.route("/temporal/status", methods=["GET"])
def get_temporal_status():
    return jsonify(cmind.temporal.get_status())

@app.route("/temporal/context", methods=["GET"])
def get_temporal_context():
    """获取最近三天记忆上下文"""
    hours = int(request.args.get("hours", 72))
    return jsonify(cmind.temporal.get_recent_context(hours))

@app.route("/temporal/context/prompt", methods=["GET"])
def get_context_prompt():
    """获取构建好的上下文提示词"""
    return jsonify({"prompt": cmind.temporal.build_context_prompt()})

@app.route("/new-window/init", methods=["GET"])
def new_window_init():
    """新窗口初始化"""
    return jsonify(cmind.new_window_init())

@app.route("/discuss", methods=["POST"])
def discuss():
    data = request.json or {}
    topic = data.get("topic", "")
    use_llm = data.get("use_llm", False)
    emotion_label = data.get("emotion", "neutral")
    session_tag = data.get("session_tag", "work")

    response = cmind.discuss(topic, use_llm=use_llm, emotion_label=emotion_label, session_tag=session_tag)

    return jsonify({
        "response": response,
        "emotion": cmind.emotion.to_dict(),
        "temporal": cmind.temporal.get_status()
    })

@app.route("/emotion/update", methods=["POST"])
def update_emotion():
    data = request.json or {}
    emotion_label = data.get("emotion", "neutral")
    cmind.temporal.update_emotion(emotion_label)
    return jsonify({"status": "ok", "emotion": emotion_label})

@app.route("/identity", methods=["GET"])
def get_identity():
    """获取身份锚定信息"""
    return jsonify(cmind.temporal.identity)

@app.route("/cells", methods=["GET"])
def get_cells():
    n = int(request.args.get("n", 8))
    return jsonify([c.to_dict() for c in cmind.ltm.top_cells(n)])

@app.route("/save", methods=["POST"])
def save_state():
    cmind.save()
    return jsonify({"status": "saved", "saved_at": datetime.now().isoformat()})

if __name__ == "__main__":
    port = 18766
    print(f"CellMind Temporal API server starting on port {port}...")
    print(f"  State dir: {TEMPORAL_STATE_DIR}")
    print(f"  LLM: {'Connected' if LLM_AVAILABLE else 'Mock mode'}")
    print(f"  时效记忆闭环: 启用")
    print(f"  三天强记忆窗口: 启用")
    app.run(host="127.0.0.1", port=port, debug=False)

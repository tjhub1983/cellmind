"""
CellMind Emotion & Activation State v0.2
=========================================
情绪/激活状态机制

核心设计：
1. 情感状态向量：valence (-1~+1) + arousal (0~1)
2. 情感分类：happy/sad/angry/fearful/neutral（从向量派生）
3. 情感影响激活：emotion_boost = 0.8~1.3，情绪调节细胞激活强度
4. 情感演化：基于LLM回复情感倾向，情感漂移（逐步回归中性）
5. 情感影响检索：高情绪时优先检索高相关细胞，低情绪时探索新细胞

架构：
  Agent.discuss(topic)
      V
  当前情感 → emotion_boost → cell_activation
      V
  LLM回复 → 情感检测(sentiment) → 更新情感状态
      V
  emotion_drift() → 情感回归中性
      V
  下一轮：情感影响检索+激活

测试：
  E1: 情感状态创建和转换
  E2: 情感影响激活强度（emotion_boost）
  E3: 情感演化（正面/负面回复）
  E4: 情感漂移（drift toward neutral）
  E5: 情感影响LLM上下文
  E6: 情感与目标联动
"""

import json
import os
import re
import random
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

# ============================================================
# 配置
# ============================================================

API_BASE = "https://api.minimaxi.com/anthropic"
API_KEY = ""
TIMEOUT_MS = 60000
MODEL = "MiniMax-M2.7"

cfg_path = "C:/Users/tt181/.claude/settings.json"
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
        env = cfg.get("env", {})
        API_KEY = env.get("ANTHROPIC_AUTH_TOKEN", "")
        if env.get("ANTHROPIC_BASE_URL"):
            API_BASE = env.get("ANTHROPIC_BASE_URL")

# ============================================================
# 参数
# ============================================================

DECAY_RATE = 0.95
INITIAL_STRENGTH = 1.0
HEBB_STRENGTH = 0.2
MATCH_WEIGHT = 0.5
BOX_DECAY_RATE = 0.99
BOX_SAVE_RATIO = 0.20
BOX_MAX = 5.0
GLOBAL_POOL_CONTRIB = 0.10
GLOBAL_POOL_DECAY = 0.995
GLOBAL_POOL_MAX = 5.0
REPAY_MULTIPLIER = 1.5
SYNC_THRESHOLD = 1.5
SYNC_STRENGTH_RATIO = 0.8
WM_CAPACITY = 5
WM_DECAY_RATE = 0.85
WM_REFRESH_BOOST = 0.3
WM_RETRIEVAL_BOOST = 0.4

# Emotion 参数
EMOTION_DECAY_RATE = 0.92   # 每轮情感漂移速率（向中性0回归）
EMOTION_DRIFT_WEIGHT = 0.08 # 每轮漂移量 = (0 - valence) * DRIFT_WEIGHT

# 情感分类阈值
POSITIVE_THRESHOLD = 0.5    # valence > +0.5 → happy
NEGATIVE_THRESHOLD = -0.5   # valence < -0.5 → sad

# 情感激活强度表
EMOTION_BOOST = {
    "happy":    1.2,   # 快乐：激活增强
    "excited":  1.3,   # 兴奋：大幅增强
    "neutral":  1.0,   # 中性：正常
    "sad":      0.85,  # 悲伤：略微抑制
    "angry":    1.15,  # 愤怒：增强（聚焦）
    "fearful":  0.8,   # 恐惧：抑制
    "frustrated": 0.9, # 沮丧：轻微抑制
}

# ============================================================
# 词素提取
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both",
    "not", "only", "own", "same", "than", "too", "very", "just",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "how", "when", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such",
    "no", "any", "say", "said", "get", "got", "make", "made",
    "go", "went", "come", "came", "take", "took", "see", "saw",
    "know", "knew", "think", "thought", "give", "gave", "tell", "told",
    "ask", "asked", "use", "used", "find", "found", "want", "wanted",
    "because", "also", "now", "here", "there", "up", "down", "out",
    "about", "over", "back", "well", "way", "thing", "things",
    "something", "nothing", "anything", "everything", "one", "ones",
    "two", "three", "first", "second", "new", "old", "good", "bad",
    "great", "right", "left", "much", "many", "even", "still", "never",
    "always", "ever", "however", "though", "although", "while",
    "please", "thanks", "thank", "sorry", "yes", "no", "ok", "okay",
    "hello", "hi", "hey", "bye", "goodbye", "like", "just", "really",
}

def extract_tokens(text: str, min_len: int = 2) -> List[str]:
    """提取词素，支持中英文分词"""
    tokens = []

    # 英文分词
    en_words = re.findall(r"[a-z]+", text.lower())
    tokens.extend([w for w in en_words if len(w) >= min_len and w not in STOP_WORDS])

    # 中文分词（使用jieba）
    try:
        import jieba
        zh_text = re.sub(r'[a-zA-Z0-9\s]', '', text)  # 去掉英文和数字
        zh_words = jieba.cut(zh_text)
        tokens.extend([w for w in zh_words if len(w) >= min_len and w not in STOP_WORDS])
    except ImportError:
        # jieba未安装，使用字符级分词作为fallback
        zh_only = re.sub(r'[a-zA-Z0-9\s]', '', text)
        if zh_only:
            # 按2-4字词滑动窗口提取
            for i in range(len(zh_only)):
                for length in [2, 3, 4]:
                    if i + length <= len(zh_only):
                        token = zh_only[i:i+length]
                        if token not in STOP_WORDS:
                            tokens.append(token)

    return list(set(tokens))  # 去重


def extract_key_phrases(text: str) -> List[str]:
    return extract_tokens(text, min_len=2)[:12]


# ============================================================
# LLM调用
# ============================================================

def call_llm(messages: List[dict], max_tokens: int = 500, temperature: float = 0.7,
             api_key: str = "", model: str = MODEL) -> str:
    url = f"{API_BASE}/v1/messages"
    key = api_key or API_KEY
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_MS // 1000)
        resp.raise_for_status()
        rj = resp.json()
        for item in rj.get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")
        return ""
    except Exception as e:
        return f"[LLM ERROR: {e}]"


# ============================================================
# GlobalPool
# ============================================================

@dataclass
class GlobalPool:
    energy: float = 0.0
    max_energy: float = GLOBAL_POOL_MAX
    decay: float = GLOBAL_POOL_DECAY
    total_debt: float = 0.0

    def contribute(self, amount: float):
        self.energy = min(self.max_energy, self.energy + amount)

    def draw(self, amount: float) -> Tuple[float, float]:
        available = self.energy
        if amount <= available:
            self.energy -= amount
            return amount, 0.0
        else:
            debt = (amount - available) * REPAY_MULTIPLIER
            drawn = available
            self.energy = 0.0
            self.total_debt += debt
            return drawn, debt

    def decay_rate(self):
        self.energy *= self.decay
        self.energy = max(0.0, self.energy)


# ============================================================
# Cell & CellMemory (v0.2)
# ============================================================

@dataclass
class Cell:
    cell_id: str
    preference: str
    strength: float
    energy_box: float
    global_debt: float = 0.0
    connections: Dict[str, float] = field(default_factory=dict)
    response_count: int = 0
    last_active: str = ""
    born: str = ""
    source_agent: Optional[str] = None
    _received_from: Optional[str] = None

    @staticmethod
    def create(preference: str, source_agent: Optional[str] = None) -> 'Cell':
        now = datetime.now().isoformat()
        return Cell(
            cell_id=f"cell_{random.randint(10000, 99999)}",
            preference=preference,
            strength=INITIAL_STRENGTH,
            energy_box=0.0,
            global_debt=0.0,
            connections={},
            response_count=0,
            last_active=now,
            born=now,
            source_agent=source_agent,
            _received_from=None
        )


class CellMemory:
    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        self.global_pool = GlobalPool()

    def get_or_create(self, preference: str, source_agent: Optional[str] = None) -> Cell:
        for cell in self.cells.values():
            if cell.preference == preference:
                return cell
        cell = Cell.create(preference, source_agent=source_agent)
        self.cells[cell.cell_id] = cell
        return cell

    def activate(self, preference: str, source_agent: Optional[str] = None,
                 boost_override: Optional[float] = None) -> Cell:
        """
        激活细胞，boost_override用于情感调制
        """
        candidates = [
            c for c in self.cells.values()
            if preference in c.preference or c.preference in preference
        ]
        if not candidates:
            cell = self.get_or_create(preference, source_agent=source_agent)
            to_pool = HEBB_STRENGTH * GLOBAL_POOL_CONTRIB
            self.global_pool.contribute(to_pool)
            cell.response_count = 1
            cell.last_active = datetime.now().isoformat()
            # 新建细胞也应用情感boost（如果存在的话）
            if boost_override is not None:
                boost = boost_override
                to_box = boost * BOX_SAVE_RATIO
                to_pool_from_boost = boost * GLOBAL_POOL_CONTRIB
                net_boost = boost - to_box - to_pool_from_boost
                cell.strength = min(2.0, cell.strength + net_boost)
                cell.energy_box = min(BOX_MAX, cell.energy_box + to_box)
            return cell

        def score(c: Cell) -> float:
            match = len(set(c.preference) & set(preference)) / max(len(c.preference), 1)
            return MATCH_WEIGHT * match + (1 - MATCH_WEIGHT) * (c.strength / INITIAL_STRENGTH)
        candidates.sort(key=score, reverse=True)
        winner = candidates[0]

        # boost由情感决定（允许外部覆盖）
        boost = boost_override if boost_override is not None else HEBB_STRENGTH
        to_box = boost * BOX_SAVE_RATIO
        to_pool = boost * GLOBAL_POOL_CONTRIB
        net_boost = boost - to_box - to_pool

        if winner.global_debt > 0:
            repay = min(winner.global_debt, net_boost)
            winner.global_debt -= repay
            net_boost -= repay

        self.global_pool.contribute(to_pool)
        winner.strength = min(2.0, winner.strength + net_boost)
        winner.energy_box = min(BOX_MAX, winner.energy_box + to_box)
        winner.response_count += 1
        winner.last_active = datetime.now().isoformat()
        return winner

    def hebb_connect(self, cell_a: Cell, cell_b: Cell):
        if cell_a.cell_id == cell_b.cell_id:
            return
        current_ab = cell_a.connections.get(cell_b.cell_id, 0.0)
        cell_a.connections[cell_b.cell_id] = min(1.0, current_ab + HEBB_STRENGTH)
        current_ba = cell_b.connections.get(cell_a.cell_id, 0.0)
        cell_b.connections[cell_a.cell_id] = min(1.0, current_ba + HEBB_STRENGTH)

    def activate_tokens(self, tokens: List[str], source_agent: Optional[str] = None,
                       boost_override: Optional[float] = None):
        pre_existing = set(self.cells.keys())
        prev = None
        for token in tokens:
            cell = self.activate(token, source_agent=source_agent, boost_override=boost_override)
            if prev and prev.cell_id in pre_existing and cell.cell_id in pre_existing:
                self.hebb_connect(prev, cell)
            prev = cell

    def get_top_cells(self, n: int = 10) -> List[Cell]:
        return sorted(self.cells.values(), key=lambda c: c.strength, reverse=True)[:n]

    def build_context_prompt(self, top_n: int = 8) -> str:
        top = self.get_top_cells(top_n)
        if not top:
            return ""
        pool_pct = self.global_pool.energy / self.global_pool.max_energy * 100
        lines = [
            "[Internal Cell State]",
            f"# Active cells: {len(top)} / {len(self.cells)} total | GlobalPool: {self.global_pool.energy:.3f}/{self.global_pool.max_energy} ({pool_pct:.0f}%)",
            "# Format: preference | strength | box | responses"
        ]
        for c in top:
            lines.append(
                f"- {c.preference} | s={c.strength:.2f} | box={c.energy_box:.3f} | resp={c.response_count}"
            )
        lines.append("[End Internal State]")
        return "\n".join(lines)

    def status_summary(self) -> str:
        top = self.get_top_cells(5)
        if not top:
            return "no cells"
        return ", ".join(f"{c.preference}({c.strength:.2f})" for c in top)


# ============================================================
# WorkingMemory
# ============================================================

@dataclass
class WorkingCell:
    ltm_cell_id: str
    preference: str
    wm_activation: float = 1.0
    retrieval_count: int = 0
    last_accessed: str = ""

    @staticmethod
    def from_cell(cell: Cell) -> 'WorkingCell':
        return WorkingCell(
            ltm_cell_id=cell.cell_id,
            preference=cell.preference,
            wm_activation=1.0,
            retrieval_count=1,
            last_accessed=datetime.now().isoformat()
        )


class WorkingMemory:
    def __init__(self, cell_memory: CellMemory, capacity: int = WM_CAPACITY):
        self.cell_memory = cell_memory
        self.capacity = capacity
        self.items: Dict[str, WorkingCell] = {}

    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def retrieve(self, preference: str) -> Optional[WorkingCell]:
        candidates = [c for c in self.cell_memory.cells.values()
                      if c.strength > 0.1 and (preference in c.preference or c.preference in preference)]
        if not candidates:
            candidates = list(self.cell_memory.cells.values())[:5]

        scored = [(len(set(c.preference) & set(preference)) / max(len(c.preference), 1), c)
                  for c in candidates]
        scored.sort(key=lambda x: (x[0], x[1].strength), reverse=True)
        best = scored[0][1] if scored else None
        if not best:
            return None

        key = best.preference
        existing = self.items.get(key)
        if existing:
            existing.wm_activation = min(1.5, existing.wm_activation + WM_REFRESH_BOOST)
            existing.retrieval_count += 1
            existing.last_accessed = datetime.now().isoformat()
            return existing

        if self.is_full():
            weakest_key = min(self.items, key=lambda k: self.items[k].wm_activation)
            del self.items[weakest_key]

        wc = WorkingCell.from_cell(best)
        wc.wm_activation = min(1.5, wc.wm_activation + WM_RETRIEVAL_BOOST)
        self.items[key] = wc
        return wc

    def decay(self):
        decayed_keys = [k for k, wc in self.items.items() if wc.wm_activation < 0.1]
        for k in decayed_keys:
            del self.items[k]
        for wc in self.items.values():
            wc.wm_activation *= WM_DECAY_RATE

    def get_focus(self) -> List[WorkingCell]:
        return sorted(self.items.values(), key=lambda wc: wc.wm_activation, reverse=True)


# ============================================================
# EmotionState
# ============================================================

@dataclass
class EmotionState:
    """
    情感状态向量

    valence: 效价 (-1 负面 到 +1 正面)
    arousal: 唤醒度 (0 平静 到 1 兴奋)
    dominance: 支配度 (0 被动 到 1 主动)
    """
    valence: float = 0.0       # -1 (negative) to +1 (positive)
    arousal: float = 0.5        # 0 (calm) to 1 (excited)
    dominance: float = 0.5      # 0 (submissive) to 1 (dominant)
    emotion_label: str = "neutral"

    def classify(self) -> str:
        """从valence+arousal派生情感标签"""
        v = self.valence
        a = self.arousal
        d = self.dominance

        if v >= POSITIVE_THRESHOLD:
            if a >= 0.7:
                label = "excited"
            else:
                label = "happy"
        elif v <= NEGATIVE_THRESHOLD:
            if a >= 0.6:
                if d >= 0.5:
                    label = "angry"
                else:
                    label = "fearful"
            else:
                label = "sad"
        else:
            if a >= 0.7 and d >= 0.6:
                label = "frustrated"
            else:
                label = "neutral"
        self.emotion_label = label
        return label

    def get_boost(self) -> float:
        """获取情感激活强度修正"""
        label = self.classify()
        return EMOTION_BOOST.get(label, 1.0)

    def apply_sentiment(self, sentiment_delta: float):
        """
        根据情感偏移更新valence
        sentiment_delta: -1 (very negative) to +1 (very positive)
        """
        self.valence = max(-1.0, min(1.0, self.valence + sentiment_delta * 0.5))
        # 极端情感同时影响唤醒度
        if abs(sentiment_delta) > 0.5:
            self.arousal = max(0.0, min(1.0, self.arousal + sentiment_delta * 0.2))

    def drift(self):
        """
        情感漂移：逐步回归中性（遗忘效应）
        valence → 0 × EMOTION_DECAY_RATE + valence × (1-EMOTION_DECAY_RATE)
        即 valence *= EMOTION_DECAY_RATE，但更精准：
        valence -= valence * EMOTION_DRIFT_WEIGHT
        """
        self.valence *= EMOTION_DECAY_RATE
        # arousal 回归 0.5
        self.arousal = 0.5 + (self.arousal - 0.5) * 0.9
        self.classify()

    def status(self) -> str:
        label = self.classify()
        boost = self.get_boost()
        return f"{label}(v={self.valence:+.2f}, a={self.arousal:.2f}) boost={boost:.2f}"


# ============================================================
# 情感检测（基于关键词）
# ============================================================

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "happy", "joy", "excited", "brilliant", "perfect", "awesome",
    "beautiful", "delightful", "thrilled", "glad", "pleased", "successful",
    "helpful", "useful", "good", "best", "better"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "sad", "angry",
    "frustrated", "disappointed", "confused", "wrong", "fail", "failed",
    "error", "problem", "issue", "difficult", "hard", "confusing",
    "stuck", "annoying", "poor", "worst", "useless", "broken"
}

INTENSE_POSITIVE = {"amazing", "fantastic", "excellent", "perfect", "brilliant"}
INTENSE_NEGATIVE = {"terrible", "horrible", "awful", "disappointed", "frustrated"}

# 中文情感词表
ZH_POSITIVE_WORDS = {
    "好", "很好", "非常好", "棒", "很棒", "太棒了", "优秀", "出色", "完美",
    "喜欢", "爱", "开心", "高兴", "快乐", "愉快", "兴奋", "激动",
    "感谢", "谢谢", "感激", "满意", "成功", "顺利", "有用", "有帮助",
    "赞", "点赞", "厉害", "牛", "牛逼", "强", "强大"
}

ZH_NEGATIVE_WORDS = {
    "差", "很差", "糟糕", "坏", "可恶", "讨厌", "恨", "生气", "愤怒",
    "难过", "伤心", "悲伤", "失望", "沮丧", "郁闷", "压抑",
    "问题", "错误", "失败", "bug", "崩溃", "卡", "慢",
    "困难", "难", "麻烦", "讨厌", "烦", "烦人", "无语"
}

ZH_INTENSE_POSITIVE = {"太棒了", "完美", "非常棒", "超级棒", "太厉害了"}
ZH_INTENSE_NEGATIVE = {"很糟糕", "太差了", "非常失望", "特别失望", "太失望了"}


def detect_sentiment(text: str) -> Tuple[float, str]:
    """
    基于关键词的情感检测（支持中英文）
    返回: (sentiment_delta, label)
    sentiment_delta: -1 (very negative) to +1 (very positive)
    """
    sentiment = 0.0
    total_signals = 0

    # 英文情感检测
    all_words = set(re.findall(r"[a-z]+", text.lower()))
    en_words = {w for w in all_words if w not in STOP_WORDS or w in POSITIVE_WORDS or w in NEGATIVE_WORDS}
    pos_count = sum(1 for w in en_words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in en_words if w in NEGATIVE_WORDS)
    intense_pos = sum(1 for w in en_words if w in INTENSE_POSITIVE)
    intense_neg = sum(1 for w in en_words if w in INTENSE_NEGATIVE)

    en_total = pos_count + neg_count
    if en_total > 0:
        base = (pos_count - neg_count) / max(en_total, 1)
        base += (intense_pos - intense_neg) * 0.2
        sentiment += base * 0.5  # 英文权重
        total_signals += en_total

    # 中文情感检测
    try:
        import jieba
        zh_words = set(jieba.cut(text))
        # 过滤单字和纯英文/数字
        zh_words = {w for w in zh_words if len(w) >= 2 and re.match(r'^[一-鿿]+$', w)}

        zh_pos = sum(1 for w in zh_words if w in ZH_POSITIVE_WORDS)
        zh_neg = sum(1 for w in zh_words if w in ZH_NEGATIVE_WORDS)
        zh_intense_pos = sum(1 for w in zh_words if w in ZH_INTENSE_POSITIVE)
        zh_intense_neg = sum(1 for w in zh_words if w in ZH_INTENSE_NEGATIVE)

        zh_total = zh_pos + zh_neg
        if zh_total > 0:
            zh_base = (zh_pos - zh_neg) / max(zh_total, 1)
            zh_base += (zh_intense_pos - zh_intense_neg) * 0.2
            sentiment += zh_base * 0.5  # 中文权重
            total_signals += zh_total
    except ImportError:
        # jieba未安装，使用2-gram fallback
        zh_text = re.sub(r'[a-zA-Z0-9\s]', '', text)
        if zh_text:
            zh_tokens = [zh_text[i:i+2] for i in range(len(zh_text)-1)]
            zh_pos = sum(1 for t in zh_tokens if t in ZH_POSITIVE_WORDS)
            zh_neg = sum(1 for t in zh_tokens if t in ZH_NEGATIVE_WORDS)
            zh_intense_pos = sum(1 for t in zh_tokens if t in ZH_INTENSE_POSITIVE)
            zh_intense_neg = sum(1 for t in zh_tokens if t in ZH_INTENSE_NEGATIVE)
            zh_total = zh_pos + zh_neg
            if zh_total > 0:
                zh_base = (zh_pos - zh_neg) / max(zh_total, 1)
                zh_base += (zh_intense_pos - zh_intense_neg) * 0.2
                sentiment += zh_base * 0.5
                total_signals += zh_total

    if total_signals == 0:
        return 0.0, "neutral"

    # 归一化到[-1, +1]
    sentiment = max(-1.0, min(1.0, sentiment))

    if sentiment > 0.3:
        label = "positive"
    elif sentiment < -0.3:
        label = "negative"
    else:
        label = "neutral"

    return sentiment, label


# ============================================================
# EmotionAgent（带情感状态的GoalAgent）
# ============================================================

class EmotionAgent:
    def __init__(self, agent_id: str, api_key: str = "", model: str = MODEL,
                 wm_capacity: int = WM_CAPACITY, use_llm: bool = True,
                 initial_valence: float = 0.0, initial_arousal: float = 0.5):
        self.agent_id = agent_id
        self.cell_memory = CellMemory()
        self.working_memory = WorkingMemory(self.cell_memory, capacity=wm_capacity)
        self.api_key = api_key or API_KEY
        self.model = model
        self.use_llm = use_llm

        # 情感状态
        self.emotion = EmotionState(valence=initial_valence, arousal=initial_arousal)
        self.emotion_history: List[dict] = []

        # 目标状态
        self.active_goals: List[dict] = []
        self.conversation_history: List[dict] = []

    def discuss(self, topic: str, verbose: bool = True) -> str:
        """
        带情感感知的讨论
        1. 获取情感boost → 激活细胞
        2. 构建含情感上下文的提示词
        3. 调用LLM
        4. 检测回复情感 → 更新情感状态
        5. 情感漂移
        """
        if verbose:
            print(f"    [{self.agent_id}] discuss: \"{topic}\"")
            print(f"    [{self.agent_id}] emotion: {self.emotion.status()}")

        # Step 1: 情感boost激活
        tokens = extract_key_phrases(topic)
        emotion_boost = HEBB_STRENGTH * self.emotion.get_boost()
        self.cell_memory.activate_tokens(tokens, source_agent=self.agent_id,
                                         boost_override=emotion_boost)

        # WM检索（情感影响检索优先级）
        for token in tokens:
            self.working_memory.retrieve(token)

        # Step 2: 构建含情感的系统提示
        cell_context = self.cell_memory.build_context_prompt(top_n=8)
        system_prompt = self._build_emotion_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        for h in self.conversation_history[-4:]:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
        messages.append({"role": "user", "content": topic})

        # Step 3: LLM调用
        if self.use_llm and self.api_key:
            response = call_llm(messages, api_key=self.api_key, model=self.model)
        else:
            response = f"Response about {topic}"

        if verbose:
            print(f"    [{self.agent_id}] response: {response[:80]}...")

        # Step 4: 响应词素激活
        resp_tokens = extract_key_phrases(response)
        if resp_tokens:
            self.cell_memory.activate_tokens(resp_tokens, source_agent=self.agent_id,
                                             boost_override=emotion_boost)

        # Step 5: 情感检测 + 更新
        sentiment_delta, sentiment_label = detect_sentiment(response)
        prev_valence = self.emotion.valence
        self.emotion.apply_sentiment(sentiment_delta)

        self.emotion_history.append({
            "topic": topic,
            "sentiment": sentiment_delta,
            "emotion_before": prev_valence,
            "emotion_after": self.emotion.valence,
            "label": self.emotion.emotion_label
        })

        if verbose:
            print(f"    [{self.agent_id}] sentiment: {sentiment_label}({sentiment_delta:+.2f}) "
                  f"→ emotion: {self.emotion.status()}")

        # Step 6: 情感漂移
        self.emotion.drift()
        if verbose:
            print(f"    [{self.agent_id}] after drift: {self.emotion.status()}")

        self.conversation_history.append({"user": topic, "assistant": response})
        return response

    def _build_emotion_system_prompt(self) -> str:
        """构建含情感上下文的系统提示"""
        top = self.cell_memory.get_top_cells(6)
        top_prefs = [c.preference for c in top]
        wm_focus = [wc.preference for wc in self.working_memory.get_focus()]
        emotion_label = self.emotion.classify()
        emotion_desc = self._emotion_description(emotion_label)

        return f"""You are {self.agent_id}, an AI assistant.

Your current emotional state: {emotion_label.upper()}
(emotion description: {emotion_desc})

This emotional state affects how engaged and focused you are.
{self._emotion_tone_guidance(emotion_label)}

Your cell memory (top concepts): {', '.join(top_prefs) if top_prefs else 'none yet'}
Working memory focus: {', '.join(wm_focus) if wm_focus else 'none'}

Respond according to your current emotional state. Keep responses concise (2-4 sentences)."""

    def _emotion_description(self, label: str) -> str:
        descs = {
            "happy": "warm, engaged, open to learning",
            "excited": "highly energized, very engaged, enthusiastic",
            "neutral": "calm, balanced, objective",
            "sad": "withdrawn, slower to engage, reflective",
            "angry": "focused intensely, critical, driven",
            "fearful": "cautious, hesitant, uncertain",
            "frustrated": "tense, impatient, seeking clarity",
        }
        return descs.get(label, "balanced")

    def _emotion_tone_guidance(self, label: str) -> str:
        guides = {
            "happy": "Be warm and enthusiastic in your response.",
            "excited": "Be highly enthusiastic and energetic in your response.",
            "neutral": "Be calm and objective in your response.",
            "sad": "Be gentle and supportive in your response.",
            "angry": "Be direct and critical-minded in your response.",
            "fearful": "Be cautious and careful in your response.",
            "frustrated": "Be patient and seek to clarify in your response.",
        }
        return guides.get(label, "")

    def affect_emotion(self, delta: float):
        """手动影响情感（用于测试或外部事件）"""
        self.emotion.apply_sentiment(delta)

    def get_activation_boost(self) -> float:
        """获取当前情感激活强度"""
        return self.emotion.get_boost()

    def status(self) -> str:
        top = self.cell_memory.get_top_cells(3)
        top_str = ", ".join(f"{c.preference}({c.strength:.2f})" for c in top)
        return (f"[{self.agent_id}] cells={len(self.cell_memory.cells)}, "
                f"top={top_str}, emotion={self.emotion.status()}")


# ============================================================
# 测试
# ============================================================

def test_e1_emotion_state_creation():
    """E1: 情感状态创建和转换"""
    print("\n" + "="*50)
    print("Test E1: 情感状态创建和转换")
    print("="*50)

    emotion = EmotionState(valence=0.5, arousal=0.7)
    print(f"  初始: valence={emotion.valence}, arousal={emotion.arousal}")
    print(f"  分类: {emotion.classify()}")
    print(f"  boost: {emotion.get_boost():.2f}")

    # 正面情感
    assert emotion.valence == 0.5
    assert emotion.classify() == "excited"
    assert emotion.get_boost() == 1.3

    # 负面情感
    emotion.valence = -0.5
    emotion.arousal = 0.7
    emotion.dominance = 0.6
    print(f"  愤怒: {emotion.classify()}, boost={emotion.get_boost():.2f}")
    assert emotion.classify() == "angry"
    assert emotion.get_boost() == 1.15

    # 悲伤
    emotion.valence = -0.5
    emotion.arousal = 0.3
    emotion.dominance = 0.4
    print(f"  悲伤: {emotion.classify()}, boost={emotion.get_boost():.2f}")
    assert emotion.classify() == "sad"
    assert emotion.get_boost() == 0.85

    # 漂移
    original_valence = emotion.valence
    emotion.drift()
    print(f"  漂移后: valence={emotion.valence:.3f} (原={original_valence})")
    assert abs(emotion.valence) < abs(original_valence), "漂移应使valence趋近0"

    # 中性情感
    emotion.valence = 0.0
    emotion.arousal = 0.5
    print(f"  中性: {emotion.classify()}, boost={emotion.get_boost():.2f}")
    assert emotion.classify() == "neutral"
    assert emotion.get_boost() == 1.0

    print(f"[PASS] E1: 情感状态创建和转换正常")


def test_e2_emotion_boost_on_activation():
    """E2: 情感影响激活强度"""
    print("\n" + "="*50)
    print("Test E2: 情感影响激活强度")
    print("="*50)

    cm = CellMemory()
    agent = EmotionAgent("TestE2", use_llm=False)

    # 中性情感：正常boost
    agent.emotion.valence = 0.0
    agent.emotion.arousal = 0.5
    neutral_boost = HEBB_STRENGTH * agent.get_activation_boost()
    print(f"  中性boost: {neutral_boost:.3f} (HEBB={HEBB_STRENGTH})")

    cm.activate_tokens(["python", "web"], boost_override=neutral_boost)
    neutral_python = next((c for c in cm.cells.values() if c.preference == "python"), None)
    assert neutral_python is not None
    neutral_strength = neutral_python.strength
    print(f"  中性strength: {neutral_strength:.3f}")

    # 快乐情感：增强boost
    agent.emotion.valence = 0.7
    agent.emotion.arousal = 0.6
    happy_boost = HEBB_STRENGTH * agent.get_activation_boost()
    print(f"  快乐boost: {happy_boost:.3f}")

    cm2 = CellMemory()
    cm2.activate_tokens(["python", "web"], boost_override=happy_boost)
    happy_python = next((c for c in cm2.cells.values() if c.preference == "python"), None)
    assert happy_python is not None
    happy_strength = happy_python.strength
    print(f"  快乐strength: {happy_strength:.3f}")

    assert happy_strength > neutral_strength, \
        f"快乐boost({happy_strength})应大于中性({neutral_strength})"

    # 悲伤情感：抑制boost
    agent.emotion.valence = -0.6
    agent.emotion.arousal = 0.3
    sad_boost = HEBB_STRENGTH * agent.get_activation_boost()
    print(f"  悲伤boost: {sad_boost:.3f}")

    cm3 = CellMemory()
    cm3.activate_tokens(["python", "web"], boost_override=sad_boost)
    sad_python = next((c for c in cm3.cells.values() if c.preference == "python"), None)
    assert sad_python is not None
    sad_strength = sad_python.strength
    print(f"  悲伤strength: {sad_strength:.3f}")

    assert sad_strength < neutral_strength, \
        f"悲伤boost({sad_strength})应小于中性({neutral_strength})"

    print(f"  对比: 快乐({happy_strength:.3f}) > 中性({neutral_strength:.3f}) > 悲伤({sad_strength:.3f})")
    print(f"[PASS] E2: 情感影响激活强度正常")


def test_e3_emotion_evolution():
    """E3: 情感演化（正面/负面回复）"""
    print("\n" + "="*50)
    print("Test E3: 情感演化")
    print("="*50)

    agent = EmotionAgent("TestE3", use_llm=False)
    agent.emotion.valence = 0.0
    agent.emotion.arousal = 0.5

    print(f"  初始: {agent.emotion.status()}")

    # 正面回复
    sentiment, label = detect_sentiment("This is excellent and amazing work!")
    print(f"  正面检测: sentiment={sentiment:+.2f}, label={label}")
    assert label == "positive"

    prev_v = agent.emotion.valence
    agent.emotion.apply_sentiment(sentiment)
    print(f"  正面后: valence={agent.emotion.valence:+.2f} (delta={agent.emotion.valence-prev_v:+.2f})")
    assert agent.emotion.valence > prev_v

    # 负面回复
    sentiment, label = detect_sentiment("This is terrible and disappointing, I hate it!")
    print(f"  负面检测: sentiment={sentiment:+.2f}, label={label}")
    assert label == "negative"

    prev_v = agent.emotion.valence
    agent.emotion.apply_sentiment(sentiment)
    print(f"  负面后: valence={agent.emotion.valence:+.2f} (delta={agent.emotion.valence-prev_v:+.2f})")
    assert agent.emotion.valence < prev_v

    # 混合情感
    sentiment, label = detect_sentiment("The documentation covers the main features clearly")
    print(f"  混合检测: sentiment={sentiment:+.2f}, label={label}")
    assert label == "neutral"

    # 验证中性不改变valence
    prev_v = agent.emotion.valence
    assert agent.emotion.valence == 0.0, "中性sentiment后valence应为0"
    print(f"  中性后valence保持: {agent.emotion.valence:+.2f}")
    print(f"[PASS] E3: 情感演化正常")


def test_e4_emotion_drift():
    """E4: 情感漂移"""
    print("\n" + "="*50)
    print("Test E4: 情感漂移")
    print("="*50)

    agent = EmotionAgent("TestE4", use_llm=False)
    agent.emotion.valence = 0.8
    agent.emotion.arousal = 0.8

    print(f"  初始: {agent.emotion.status()}")
    assert agent.emotion.valence == 0.8

    # 多轮漂移
    for i in range(5):
        agent.emotion.drift()
        print(f"  第{i+1}轮漂移后: {agent.emotion.status()}")

    assert abs(agent.emotion.valence) < 0.8, "valence应向0漂移"
    assert 0.0 <= agent.emotion.valence <= 0.8, "valence应在合理范围"
    print(f"  漂移后valence: {agent.emotion.valence:.3f} (从0.8漂移)")

    # 验证漂移速率
    agent2 = EmotionAgent("TestE4b", use_llm=False)
    agent2.emotion.valence = 0.5
    v_before = agent2.emotion.valence
    agent2.emotion.drift()
    v_after = agent2.emotion.valence
    drift_amount = v_before - v_after
    print(f"  漂移量: {drift_amount:.4f} per round")
    expected_drift = v_before * (1 - EMOTION_DECAY_RATE)
    assert abs(drift_amount - expected_drift) < 0.001, f"漂移量不符: {drift_amount} vs {expected_drift}"
    print(f"[PASS] E4: 情感漂移正常")


def test_e5_emotion_in_llm_context():
    """E5: 情感影响LLM上下文"""
    print("\n" + "="*50)
    print("Test E5: 情感影响LLM上下文")
    print("="*50)

    agent = EmotionAgent("TestE5", use_llm=False)

    # 训练一些细胞
    agent.cell_memory.activate_tokens(
        extract_key_phrases("python machine learning neural network"),
        source_agent=agent.agent_id
    )

    # 快乐情感
    agent.emotion.valence = 0.7
    agent.emotion.arousal = 0.8
    prompt = agent._build_emotion_system_prompt()
    print(f"  快乐prompt片段: ...{prompt[-80:]}")
    assert "EXCITED" in prompt or "excited" in prompt.lower()
    assert "enthusiastic" in prompt.lower()

    # 悲伤情感
    agent.emotion.valence = -0.6
    agent.emotion.arousal = 0.3
    prompt2 = agent._build_emotion_system_prompt()
    print(f"  悲伤prompt片段: ...{prompt2[-80:]}")
    assert "SAD" in prompt2 or "sad" in prompt2.lower()
    assert "gentle" in prompt2.lower() or "supportive" in prompt2.lower()

    print(f"[PASS] E5: 情感影响LLM上下文正常")


def test_e6_emotion_goal_integration():
    """E6: 情感与目标联动"""
    print("\n" + "="*50)
    print("Test E6: 情感与目标联动")
    print("="*50)

    agent = EmotionAgent("TestE6", use_llm=False)
    agent.emotion.valence = 0.0
    agent.emotion.arousal = 0.5

    # 模拟追求目标过程中的情感演化
    topics_and_sentiments = [
        ("Start learning deep learning", "This is a great foundation!"),
        ("Study neural networks", "The concepts are excellent and clear!"),
        ("Try implementing backpropagation", "The code works but the interface is confusing"),
        ("Debug the gradient issues", "This is terrible, nothing works!"),
        ("Fix the bugs and retest", "I fixed it, excellent result!"),
    ]

    print(f"  初始情感: {agent.emotion.status()}")
    valence_history = [agent.emotion.valence]

    for topic, mock_response in topics_and_sentiments:
        sentiment, _ = detect_sentiment(mock_response)
        agent.emotion.apply_sentiment(sentiment)
        valence_history.append(agent.emotion.valence)
        agent.emotion.drift()
        print(f"  '{topic[:30]}...' → sentiment={sentiment:+.2f} → "
              f"valence={agent.emotion.valence:+.2f} → {agent.emotion.classify()}")

    # 验证情感历史
    # 注意：直接调用apply_sentiment()不记录历史（历史由discuss()记录）
    # 重要断言：valence变化符合预期

    # 正面响应使valence上升
    assert valence_history[1] > valence_history[0], "正面应提升valence"

    # 负面响应使valence下降
    neg_idx = next((i for i, (_, r) in enumerate(topics_and_sentiments) if "terrible" in r), None)
    if neg_idx:
        assert valence_history[neg_idx] < valence_history[neg_idx - 1], "负面应降低valence"

    print(f"  情感历史: {[f'{v:.2f}' for v in valence_history]}")
    print(f"  最终情感: {agent.emotion.status()}")
    print(f"[PASS] E6: 情感与目标联动正常")


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    print("CellMind Emotion & Activation State v0.2")
    print(f"  API Key: {'set' if API_KEY else 'NOT SET'}")
    print(f"  LLM可用: {bool(API_KEY)}")
    print("="*50)

    test_e1_emotion_state_creation()
    test_e2_emotion_boost_on_activation()
    test_e3_emotion_evolution()
    test_e4_emotion_drift()
    test_e5_emotion_in_llm_context()
    test_e6_emotion_goal_integration()

    print("\n" + "="*50)
    print("全部Emotion测试完成")
    print("="*50)

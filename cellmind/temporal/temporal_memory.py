#!/usr/bin/env python3
"""
CellMind v0.2 — 时效连贯记忆闭环
================================
三天内对话统一入库，时间戳 + 会话标签 + 情绪标签
汇入细胞池，走同一套流水线：特征提取 → 分类筛选 → 权重打分 → 记忆固化/自然淡忘

核心规则：
1. 三天内所有对话记录入库，不脱节
2. 新开对话窗口，第一步锚定身份「我是CellMind的小贝」
3. 第二步自动挂载最近三天完整记忆上下文
4. 人格情感一致性全程锁定
"""
import os, sys, json, random, re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# ── 时效配置 ──────────────────────────────────────────────────────────────────
class TemporalConfig:
    """时效记忆配置"""
    # 三天强记忆衔接期（毫秒）
    STRONG_MEMORY_WINDOW_MS = 3 * 24 * 60 * 60 * 1000  # 72小时

    # 淡忘降权阈值
    FORGET_THRESHOLD_DAYS = 3      # 3天后开始淡忘
    PERMANENT_IMPORTANCE_THRESHOLD = 0.85  # 权重>0.85自动永久保留

    # 入库流水线权重
    FEATURE_EXTRACT_BOOST = 0.3     # 特征提取激活权重
    CLASSIFY_BOOST = 0.25          # 分类筛选激活权重
    WEIGHT_SCORE_BOOST = 0.4       # 权重打分激活权重

    # 情感标签
    EMOTION_LABELS = ["excited", "happy", "neutral", "frustrated", "sad", "angry", "fearful"]

    # 会话标签
    SESSION_TAGS = ["work", "casual", "creative", "technical", "personal", "planning"]

@dataclass
class TemporalRecord:
    """时效记忆记录"""
    record_id: str
    timestamp: str                    # ISO格式时间戳
    role: str                         # "user" | "assistant"
    content: str                      # 对话内容
    session_tag: str                  # 会话标签
    emotion_label: str               # 情绪标签
    tokens: List[str] = field(default_factory=list)      # 提取的词素
    importance_score: float = 0.5    # 重要性打分
    cell_ids: List[str] = field(default_factory=list)    # 关联的细胞ID
    is_permanent: bool = False        # 是否永久保留

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> 'TemporalRecord':
        return TemporalRecord(**d)

    @property
    def age_hours(self) -> float:
        """计算记录年龄（小时）"""
        created = datetime.fromisoformat(self.timestamp)
        age = datetime.now() - created
        return age.total_seconds() / 3600

    @property
    def age_days(self) -> float:
        """计算记录年龄（天）"""
        return self.age_hours / 24

    @property
    def is_within_strong_window(self) -> bool:
        """是否在三天强记忆衔接期内"""
        return self.age_hours <= (TemporalConfig.STRONG_MEMORY_WINDOW_MS / 1000 / 3600)


class TemporalMemory:
    """
    时效记忆层
    ==========
    管理三天内所有对话记录，统一入库，走流水线处理
    """

    def __init__(self, state_dir: str = "~/.claude/cmind/"):
        self.state_dir = os.path.expanduser(state_dir)
        self.records: List[TemporalRecord] = []
        self.sessions: Dict[str, List[str]] = defaultdict(list)  # session_id -> record_ids
        self.current_session_id: str = self._new_session_id()

        # 身份锚定
        self.identity = {
            "name": "CellMind小贝",
            "personality": "类脑AI，记忆持久，情感连贯",
            "core_traits": ["长期记忆", "独立思考", "自主决策", "情感一致"]
        }

        # 情感状态锁定
        self.emotion_lock = {
            "current_state": "neutral",
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5,
            "history": []
        }

        self._load()

    def _new_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"

    # ── 入库流水线 ────────────────────────────────────────────────────────────

    def add_record(
        self,
        content: str,
        role: str,
        emotion_label: str = "neutral",
        session_tag: str = "work"
    ) -> TemporalRecord:
        """
        添加记录 → 走流水线
        特征提取 → 分类筛选 → 权重打分 → 记忆固化
        """

        # 1. 特征提取
        tokens = self._extract_features(content)

        # 2. 分类筛选
        importance = self._classify_importance(content, tokens, emotion_label)

        # 3. 创建记录
        record = TemporalRecord(
            record_id=f"rec_{datetime.now().timestamp()}_{random.randint(1000,9999)}",
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            session_tag=session_tag,
            emotion_label=emotion_label,
            tokens=tokens,
            importance_score=importance,
            is_permanent=importance >= TemporalConfig.PERMANENT_IMPORTANCE_THRESHOLD
        )

        # 4. 汇入记录池
        self.records.append(record)
        self.sessions[self.current_session_id].append(record.record_id)

        # 5. 持久化
        self._save()

        return record

    def _extract_features(self, content: str) -> List[str]:
        """特征提取：从内容中提取关键词素"""
        # 简单分词 + 停用词过滤
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "and",
            "or", "but", "if", "not", "no", "so", "as", "it", "its", "they",
            "我", "你", "他", "她", "的", "了", "在", "是", "有", "和", "就",
            "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要",
            "啊", "哦", "嗯", "呢", "吧", "吗", "呀", "哈"
        }

        words = content.lower()
        words = re.sub(r'[^\w\s一-鿿]', ' ', words)
        tokens = [w.strip() for w in words.split() if w.strip() and len(w.strip()) > 1]
        tokens = [t for t in tokens if t not in stop_words]

        # 提取关键实体（简单模式）
        key_patterns = [
            r'(?:项目|任务|功能|模块|接口|设计|实现|测试|部署|bug|修复)',
            r'(?:CellMind|V8|ProtoCell|记忆|细胞|记忆闭环)',
            r'(?:桃桃|小贝|张一鸣|马斯克|王坚)',
        ]

        for pattern in key_patterns:
            matches = re.findall(pattern, content)
            tokens.extend(matches)

        return list(set(tokens))[:20]  # 最多20个词素

    def _classify_importance(
        self,
        content: str,
        tokens: List[str],
        emotion_label: str
    ) -> float:
        """分类筛选 + 权重打分"""
        score = 0.5  # 基础分

        # 情绪权重
        emotion_weights = {
            "excited": 0.15, "happy": 0.1, "angry": 0.1,
            "frustrated": -0.05, "sad": -0.05, "fearful": -0.05
        }
        score += emotion_weights.get(emotion_label, 0)

        # 内容重要性检测
        important_keywords = [
            "重要", "关键", "核心", "必须", "紧急", "优先", "决定", "确认",
            "项目", "任务", "目标", "deadline", "交付", "完成",
            "bug", "修复", "问题", "错误", "issue"
        ]
        for kw in important_keywords:
            if kw.lower() in content.lower():
                score += 0.05

        # 词素密度
        if len(tokens) > 10:
            score += 0.1

        return min(1.0, max(0.0, score))

    # ── 淡忘机制 ──────────────────────────────────────────────────────────────

    def apply_decay(self) -> List[TemporalRecord]:
        """
        应用淡忘机制
        - 超出三天的内容慢慢降权重
        - 重要人物、项目、情绪记忆自动升权重
        """
        decayed = []

        for record in self.records:
            age_days = record.age_days

            # 三天内强记忆期，不淡忘
            if age_days <= TemporalConfig.FORGET_THRESHOLD_DAYS:
                continue

            # 超出三天，按比例降权重
            decay_factor = 0.95 ** (age_days - TemporalConfig.FORGET_THRESHOLD_DAYS)
            old_score = record.importance_score
            record.importance_score *= decay_factor

            # 重要内容自动保护
            if self._is_important_content(record):
                record.importance_score = max(record.importance_score, 0.6)
                record.is_permanent = True

            if old_score != record.importance_score:
                decayed.append(record)

        # 移除权重过低的记录（保留永久记录）
        self.records = [
            r for r in self.records
            if r.importance_score >= 0.1 or r.is_permanent
        ]

        self._save()
        return decayed

    def _is_important_content(self, record: TemporalRecord) -> bool:
        """判断内容是否重要到需要永久保留"""
        important_patterns = [
            r'(?:桃桃|小贝|用户)',
            r'(?:CellMind|记忆闭环|项目)',
            r'(?:重要|决定|确认|完成)',
        ]
        content_lower = record.content.lower()

        for pattern in important_patterns:
            if re.search(pattern, content_lower):
                return True

        # 情感强烈的内容
        if record.emotion_label in ["excited", "angry", "happy"]:
            return True

        return False

    # ── 上下文加载 ─────────────────────────────────────────────────────────────

    def get_recent_context(self, hours: int = 72) -> Dict[str, Any]:
        """
        获取最近N小时的记忆上下文
        用于新窗口自动挂载
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            r for r in self.records
            if datetime.fromisoformat(r.timestamp) >= cutoff
        ]

        # 按时间排序
        recent.sort(key=lambda r: r.timestamp)

        return {
            "records": [r.to_dict() for r in recent],
            "record_count": len(recent),
            "time_range": {
                "from": recent[0].timestamp if recent else None,
                "to": recent[-1].timestamp if recent else None
            },
            "identity": self.identity,
            "emotion": self.emotion_lock,
            "sessions": self.sessions
        }

    def build_context_prompt(self) -> str:
        """
        构建上下文提示词
        用于注入到LLM
        """
        recent = self.get_recent_context(hours=72)
        records = recent["records"]

        if not records:
            return ""

        # 构建时间线
        timeline = []
        for r in records[-10:]:  # 最近10条
            role_label = "用户" if r["role"] == "user" else "我"
            emotion = f"[{r['emotion_label']}]" if r['emotion_label'] != 'neutral' else ""
            timeline.append(f"{role_label}{emotion}: {r['content'][:100]}...")

        prompt = f"""
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
        return prompt

    # ── 情感更新 ──────────────────────────────────────────────────────────────

    def update_emotion(self, emotion_label: str):
        """更新情感状态（锁定）"""
        self.emotion_lock["current_state"] = emotion_label
        self.emotion_lock["history"].append(f"{emotion_label}:{datetime.now().isoformat()}")

        # 保留最近20条情感历史
        if len(self.emotion_lock["history"]) > 20:
            self.emotion_lock["history"] = self.emotion_lock["history"][-20:]

        self._save()

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def _state_path(self) -> str:
        return os.path.join(self.state_dir, "temporal_memory.json")

    def _save(self):
        os.makedirs(self.state_dir, exist_ok=True)
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

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """获取时效记忆状态"""
        now = datetime.now()
        cutoff_3days = now - timedelta(days=3)

        recent_records = [
            r for r in self.records
            if datetime.fromisoformat(r.timestamp) >= cutoff_3days
        ]

        permanent_records = [r for r in self.records if r.is_permanent]

        return {
            "total_records": len(self.records),
            "records_3days": len(recent_records),
            "permanent_records": len(permanent_records),
            "sessions": len(self.sessions),
            "current_session": self.current_session_id,
            "identity": self.identity,
            "emotion": self.emotion_lock["current_state"],
            "strong_window_active": True
        }


# ── 单元测试 ──────────────────────────────────────────────────────────────────
def run_temporal_memory_tests():
    print("=" * 60)
    print("CellMind v0.2 — 时效连贯记忆闭环测试")
    print("=" * 60)

    tm = TemporalMemory(state_dir="/tmp/cmind_temporal_test")

    # ── Test 1: 记录入库流水线 ──────────────────────────────────────────────
    print("\n[Test 1] 记录入库流水线")
    r1 = tm.add_record(
        content="我们今天要完成CellMind时效记忆闭环的架构设计",
        role="user",
        emotion_label="excited",
        session_tag="technical"
    )
    print(f"  记录ID: {r1.record_id}")
    print(f"  提取词素: {r1.tokens}")
    print(f"  重要性: {r1.importance_score:.2f}")
    assert len(r1.tokens) > 0, "特征提取应该提取到词素"
    print("  PASS")

    # ── Test 2: 情感锁定 ─────────────────────────────────────────────────────
    print("\n[Test 2] 情感状态锁定")
    tm.update_emotion("happy")
    assert tm.emotion_lock["current_state"] == "happy"
    assert len(tm.emotion_lock["history"]) > 0
    print(f"  当前情感: {tm.emotion_lock['current_state']}")
    print(f"  情感历史: {tm.emotion_lock['history']}")
    print("  PASS")

    # ── Test 3: 上下文构建 ──────────────────────────────────────────────────
    print("\n[Test 3] 上下文提示词构建")
    tm.add_record("这是助手回复的内容", role="assistant", emotion_label="neutral")
    prompt = tm.build_context_prompt()
    assert "CellMind身份锚定" in prompt
    assert "72小时内" in prompt
    print(f"  提示词长度: {len(prompt)} 字符")
    print("  PASS")

    # ── Test 4: 状态查询 ────────────────────────────────────────────────────
    print("\n[Test 4] 状态查询")
    status = tm.get_status()
    print(f"  总记录: {status['total_records']}")
    print(f"  3天内: {status['records_3days']}")
    print(f"  永久记录: {status['permanent_records']}")
    print(f"  强记忆窗口: {status['strong_window_active']}")
    assert status["records_3days"] >= 2, "应该至少有2条记录在3天内"
    print("  PASS")

    # ── Test 5: 会话分离 ────────────────────────────────────────────────────
    print("\n[Test 5] 会话分离")
    old_session = tm.current_session_id
    tm.current_session_id = tm._new_session_id()
    tm.add_record("新会话的第一条消息", role="user")
    assert tm.current_session_id != old_session
    assert len(tm.sessions[tm.current_session_id]) == 1
    print(f"  旧会话: {old_session}")
    print(f"  新会话: {tm.current_session_id}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All 5 tests PASSED ✓")
    print("=" * 60)
    print("\n时效记忆闭环规则：")
    print("  1. 三天内对话统一入库，走流水线")
    print("  2. 三天强记忆衔接期，不淡忘")
    print("  3. 新窗口自动挂载72小时记忆上下文")
    print("  4. 身份锚定 + 情感锁定全程生效")


if __name__ == "__main__":
    run_temporal_memory_tests()

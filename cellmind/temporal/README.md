# CellMind v2.1 — 时效连贯记忆闭环

> **版本号**: 2.1.0
> **发布日期**: 2026-05-07
> **版本类型**: 单例分支，独立打包
> **负责人交接**: 朱旭 → 小贝

---

## 版本说明

CellMind v2.1 是时效连贯记忆闭环的独立封装版本，从CellMind v0.2核心分支独立出来，不影响原有架构。

### 核心功能
- 三天时效上下文（72小时强记忆窗口）
- 并入CellMind细胞池记忆流转
- 新窗口无缝接续
- 身份锚定 + 情感锁定
- 统一入库流水线

---

## 目录结构

```
CellMind_v2.1_时效记忆闭环/
├── VERSION.py           # 版本信息
├── temporal_memory.py  # 时效记忆层核心
├── temporal_api.py     # Flask API服务器（端口18766）
├── frontend_api.js     # 前端对接模块
├── test_temporal.py    # 完整测试套件
├── launch.py          # 快速启动脚本
└── README.md          # 本文件
```

---

## 快速启动

### 方式1: 快速启动脚本
```bash
cd E:/CellMind_桃桃总控台/CellMind_v2.1_时效记忆闭环
python launch.py
```

### 方式2: 直接启动API服务器
```bash
cd E:/CellMind_桃桃总控台/CellMind_v2.1_时效记忆闭环
python temporal_api.py
```

### 方式3: 集成到现有CellMind
```python
from temporal_memory import TemporalMemory

# 时效记忆层初始化
tm = TemporalMemory()

# 对话入库
tm.add_record("用户消息", "user", "neutral", "work")

# 获取上下文
prompt = tm.build_context_prompt()
```

---

## API接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/new-window/init` | GET | 新窗口初始化 |
| `/temporal/context` | GET | 获取最近N小时记忆上下文 |
| `/temporal/context/prompt` | GET | 获取上下文提示词 |
| `/discuss` | POST | 对话 + 入库 |
| `/emotion/update` | POST | 更新情感状态 |
| `/identity` | GET | 获取身份锚定信息 |
| `/status` | GET | 系统状态 |
| `/save` | POST | 持久化保存 |

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STRONG_MEMORY_WINDOW_MS` | 259200000 (72小时) | 三天强记忆衔接期 |
| `FORGET_THRESHOLD_DAYS` | 3 | 开始淡忘的天数 |
| `PERMANENT_IMPORTANCE_THRESHOLD` | 0.85 | 永久保留阈值 |

---

## 数据存储

默认存储路径: `~/.claude/cmind/`
- `temporal_memory.json` - 时效记忆数据
- `cmind_temporal_state.json` - CellMind状态

---

## 测试

```bash
cd E:/CellMind_桃桃总控台/CellMind_v2.1_时效记忆闭环
python test_temporal.py
```

---

## 交接清单

**朱旭 → 小贝 交接内容：**

1. [x] 源码包: `CellMind_v2.1_时效记忆闭环/`
2. [x] 版本说明: `VERSION.py`
3. [x] API文档: `README.md`
4. [x] 测试脚本: `test_temporal.py`
5. [x] 前端模块: `frontend_api.js`
6. [x] 快速启动: `launch.py`

**小贝负责：**
- 后续版本迭代
- 部署维护
- bug修复
- 与CellMind v0.2的集成对接

---

## 变更日志

```
v2.1.0 (2026-05-07)
- 首次发布
- 三天时效上下文
- 身份锚定 + 情感锁定
- 新窗口无缝接续
- 统一入库流水线
```

---

## 联系方式

如有问题，请查阅：
- 源码注释: `temporal_memory.py`
- 测试用例: `test_temporal.py`
- API文档: `temporal_api.py`

---

*CellMind v2.1 | 朱旭 整理 | 2026-05-07*

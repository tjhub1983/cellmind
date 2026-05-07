#!/usr/bin/env python3
"""
CellMind Temporal - 时效连贯记忆闭环
======================================
Version: 2.1.0
License: Apache 2.0
Repository: https://github.com/tjhub1983/cellmind

让AI拥有72小时时效记忆窗口，实现跨会话的上下文连贯性。

核心功能：
- 三天时效上下文（72小时强记忆窗口）
- 并入CellMind细胞池记忆流转
- 新窗口无缝接续
- 身份锚定 + 情感锁定
- 统一入库流水线
"""

VERSION = "2.1.0"
VERSION_NAME = "CellMind Temporal v2.1 时效记忆闭环"

__version__ = VERSION
__all__ = ["TemporalMemory", "TemporalRecord", "TemporalConfig"]
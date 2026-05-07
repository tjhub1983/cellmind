/**
 * CellMind v0.2 — 前端时效记忆闭环API
 * =====================================
 * 对接像素风UI + 时效记忆闭环API (端口18766)
 *
 * 新窗口初始化流程：
 * 1. 调用 /new-window/init 获取身份锚定 + 记忆上下文
 * 2. 自动注入到系统提示词
 * 3. 所有对话自动入库，带时间戳 + 会话标签 + 情绪标签
 */

const CONFIG = {
  // API配置
  TEMPORAL_API_BASE: 'http://127.0.0.1:18766',
  ANTHROPIC_BASE: 'https://api.anthropic.com/v1',
  MODEL: 'claude-sonnet-4-20250514',

  // 时效配置
  STRONG_MEMORY_WINDOW_HOURS: 72,  // 3天强记忆衔接期
  AUTO_SAVE_INTERVAL: 30000,       // 30秒自动保存

  // 截断配置
  MAX_TOKENS: 200000,
  AUTO_TRUNCATE_THRESHOLD: 0.9,

  // 身份锚定
  IDENTITY: {
    name: 'CellMind小贝',
    personality: '类脑AI，记忆持久，情感连贯',
    core_traits: ['长期记忆', '独立思考', '自主决策', '情感一致']
  },

  // 情感标签
  EMOTION_LABELS: ['neutral', 'happy', 'excited', 'frustrated', 'sad', 'angry', 'fearful'],

  // 会话标签
  SESSION_TAGS: ['work', 'casual', 'creative', 'technical', 'personal', 'planning'],

  // 模拟模式
  MOCK_MODE: true,
  MOCK_DELAY: 1000
};

// ── 全局状态 ──────────────────────────────────────────────────────────────────
let temporalState = {
  initialized: false,
  sessionId: null,
  identityAnchor: null,
  contextPrompt: null,
  emotionState: 'neutral',
  recentRecords: [],
  totalRecords: 0
};

// ── 时效记忆API ───────────────────────────────────────────────────────────────

/**
 * 初始化新窗口
 * 1. 锚定身份
 * 2. 挂载记忆上下文
 */
async function initNewWindow() {
  try {
    const resp = await fetch(`${CONFIG.TEMPORAL_API_BASE}/new-window/init`);
    if (resp.ok) {
      const data = await resp.json();
      temporalState.initialized = true;
      temporalState.identityAnchor = data.identity_anchor;
      temporalState.contextPrompt = data.context_prompt;
      temporalState.emotionState = data.emotion_state?.state || 'neutral';
      temporalState.recentRecords = data.recent_records?.records || [];
      temporalState.totalRecords = data.recent_records?.record_count || 0;

      console.log('[CellMind] 新窗口初始化完成');
      console.log(`  身份: ${temporalState.identityAnchor}`);
      console.log(`  记忆上下文: ${temporalState.recentRecords.length}条记录`);

      return {
        identity: temporalState.identityAnchor,
        context: temporalState.contextPrompt,
        emotion: temporalState.emotionState,
        records: temporalState.recentRecords
      };
    }
  } catch (e) {
    console.log('[CellMind] 时效API不可用，使用本地模式');
  }

  // 回退：返回默认身份锚定
  return {
    identity: `我是${CONFIG.IDENTITY.name}，${CONFIG.IDENTITY.personality}。`,
    context: '',
    emotion: 'neutral',
    records: []
  };
}

/**
 * 发送消息（带时效记忆）
 */
async function sendMessageWithTemporal(userMessage, options = {}) {
  const {
    useLLM = false,
    emotion = 'neutral',
    sessionTag = 'work',
    apiKey = null
  } = options;

  // 1. 先调用时效API入库
  try {
    await fetch(`${CONFIG.TEMPORAL_API_BASE}/discuss`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic: userMessage,
        use_llm: false,  // 前端直接调用，不走API
        emotion: emotion,
        session_tag: sessionTag
      })
    });
  } catch (e) {
    console.log('[CellMind] 入库失败，继续对话');
  }

  // 2. 构建上下文增强的提示词
  const enhancedPrompt = buildEnhancedPrompt(userMessage);

  // 3. 调用LLM
  if (useLLM && apiKey) {
    return await callAnthropicWithContext(enhancedPrompt, apiKey);
  } else {
    return await mockResponse(enhancedPrompt);
  }
}

/**
 * 构建增强提示词
 */
function buildEnhancedPrompt(userMessage) {
  const identity = temporalState.identityAnchor ||
    `我是${CONFIG.IDENTITY.name}，${CONFIG.IDENTITY.personality}。`;

  let context = '';
  if (temporalState.contextPrompt) {
    context = temporalState.contextPrompt;
  } else if (temporalState.recentRecords.length > 0) {
    const recentMessages = temporalState.recentRecords.slice(-6).map(r => {
      const role = r.role === 'user' ? '用户' : '我';
      const emotion = r.emotion_label !== 'neutral' ? `[${r.emotion_label}]` : '';
      return `${role}${emotion}: ${r.content.substring(0, 80)}...`;
    }).join('\n');

    context = `
【最近对话（72小时）】
${recentMessages}
    `.trim();
  }

  return `
${identity}

${context ? `【记忆上下文】\n${context}\n` : ''}
【当前情感状态】
状态: ${temporalState.emotionState}

现在用户说: ${userMessage}

请以CellMind小贝的身份回复，保持记忆连贯和情感一致。
`.trim();
}

/**
 * 调用Anthropic API
 */
async function callAnthropicWithContext(prompt, apiKey) {
  try {
    const resp = await fetch(`${CONFIG.ANTHROPIC_BASE}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true'
      },
      body: JSON.stringify({
        model: CONFIG.MODEL,
        max_tokens: 4096,
        messages: [{ role: 'user', content: prompt }]
      })
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error?.message || `API错误: ${resp.status}`);
    }

    const data = await resp.json();

    // 入库助手回复
    try {
      await fetch(`${CONFIG.TEMPORAL_API_BASE}/discuss`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: data.content[0].text,
          use_llm: false,
          emotion: 'neutral',
          session_tag: 'work'
        })
      });
    } catch (e) {}

    return data.content[0].text;
  } catch (error) {
    console.error('[CellMind] API调用失败:', error);
    return await mockResponse(prompt);
  }
}

/**
 * 模拟响应
 */
async function mockResponse(prompt) {
  await new Promise(resolve => setTimeout(resolve, CONFIG.MOCK_DELAY));

  const responses = [
    `我理解了。\n\n作为CellMind小贝，我的记忆系统正在记录这个对话。\n\n当前情感状态: ${temporalState.emotionState}\n记忆上下文: ${temporalState.recentRecords.length}条相关记录\n\n请问还有什么需要我帮助的？`,
    `嗯，让我想想。\n\n基于我们的对话历史，我已经建立了相关的记忆连接。\n\n保持人格一致性，我会持续跟进这个话题。`,
    `好的，我收到了。\n\n【CellMind 记忆更新】\n- 词素激活中...\n- 情感状态同步\n- 记忆固化完成\n\n有什么我可以继续帮助的吗？`
  ];

  return responses[Math.floor(Math.random() * responses.length)];
}

// ── 情感更新 ─────────────────────────────────────────────────────────────────

/**
 * 更新情感状态
 */
async function updateEmotion(emotionLabel) {
  temporalState.emotionState = emotionLabel;

  try {
    await fetch(`${CONFIG.TEMPORAL_API_BASE}/emotion/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emotion: emotionLabel })
    });
  } catch (e) {
    console.log('[CellMind] 情感更新入库失败');
  }
}

// ── 上下文状态 ───────────────────────────────────────────────────────────────

/**
 * 获取时效记忆状态
 */
async function getTemporalStatus() {
  try {
    const resp = await fetch(`${CONFIG.TEMPORAL_API_BASE}/temporal/status`);
    if (resp.ok) {
      return await resp.json();
    }
  } catch (e) {}

  return {
    total_records: temporalState.totalRecords,
    records_3days: temporalState.recentRecords.length,
    strong_window_active: true,
    identity: CONFIG.IDENTITY
  };
}

/**
 * 获取记忆上下文提示词
 */
async function getContextPrompt() {
  try {
    const resp = await fetch(`${CONFIG.TEMPORAL_API_BASE}/temporal/context/prompt`);
    if (resp.ok) {
      const data = await resp.json();
      temporalState.contextPrompt = data.prompt;
      return data.prompt;
    }
  } catch (e) {}

  return temporalState.contextPrompt || '';
}

// ── 前端UI集成 ────────────────────────────────────────────────────────────────

/**
 * 初始化UI（增强版）
 */
async function initUI() {
  // 1. 初始化新窗口
  const initData = await initNewWindow();

  // 2. 显示欢迎信息
  const container = document.getElementById('chatContainer');
  if (container) {
    container.innerHTML = `
      <div class="welcome">
        <div class="welcome-title">◈ CELLMIND 时效记忆闭环 ◈</div>
        <div class="welcome-text">
          身份锚定: ${initData.identity.substring(0, 30)}...<br>
          记忆上下文: ${initData.records.length}条记录加载中<br>
          情感状态: ${initData.emotion}<br><br>
          输入消息开始对话<br>
          三天内对话全程连贯，不许断片、不许重置
        </div>
      </div>
    `;
  }

  // 3. 更新状态显示
  updateTemporalStatus();

  // 4. 启动自动保存
  setInterval(autoSave, CONFIG.AUTO_SAVE_INTERVAL);

  return initData;
}

/**
 * 更新状态显示
 */
async function updateTemporalStatus() {
  const status = await getTemporalStatus();

  // 更新记忆状态指示灯
  const memoryDot = document.getElementById('memoryStatus');
  if (memoryDot) {
    if (status.records_3days > 0) {
      memoryDot.classList.remove('error');
    } else {
      memoryDot.classList.add('warning');
    }
  }

  // 更新上下文状态
  const contextDot = document.getElementById('contextStatus');
  if (contextDot) {
    if (status.strong_window_active) {
      contextDot.classList.remove('error');
    }
  }

  console.log('[CellMind] 时效状态:', status);
}

/**
 * 自动保存
 */
async function autoSave() {
  try {
    await fetch(`${CONFIG.TEMPORAL_API_BASE}/save`, {
      method: 'POST'
    });
    console.log('[CellMind] 自动保存完成');
  } catch (e) {
    console.log('[CellMind] 自动保存失败');
  }
}

/**
 * 发送消息（UI集成版）
 */
async function sendMessageUI() {
  const input = document.getElementById('userInput');
  const text = input.value.trim();

  if (!text || isLoading) return;

  addMessage('user', text);
  input.value = '';
  isLoading = true;
  updateSendButton();

  const loadingEl = addLoadingMessage();

  try {
    // 获取情感标签
    const emotionSelect = document.getElementById('emotionSelect');
    const emotion = emotionSelect ? emotionSelect.value : 'neutral';

    const response = await sendMessageWithTemporal(text, {
      useLLM: true,
      emotion: emotion,
      sessionTag: 'work',
      apiKey: localStorage.getItem('ANTHROPIC_API_KEY')
    });

    loadingEl.remove();
    addMessage('assistant', response);

    // 更新情感状态
    temporalState.emotionState = detectEmotion(response);
    await updateEmotion(temporalState.emotionState);

  } catch (error) {
    loadingEl.remove();
    addMessage('assistant', `[错误] ${error.message}`);
  }

  isLoading = false;
  updateSendButton();
}

/**
 * 检测情感
 */
function detectEmotion(text) {
  const positive = ['好', '棒', '厉害', '赞', '不错', '谢谢', 'happy', 'great'];
  const negative = ['不', '没', '难', '问题', '错误', 'bug', '麻烦'];

  let score = 0;
  for (const p of positive) {
    if (text.includes(p)) score++;
  }
  for (const n of negative) {
    if (text.includes(n)) score--;
  }

  if (score > 1) return 'happy';
  if (score < -1) return 'frustrated';
  return 'neutral';
}

// ── 导出API ──────────────────────────────────────────────────────────────────

window.CellMindTemporal = {
  init: initNewWindow,
  initUI: initUI,
  send: sendMessageWithTemporal,
  sendUI: sendMessageUI,
  updateEmotion: updateEmotion,
  getStatus: getTemporalStatus,
  getContext: getContextPrompt,
  state: () => temporalState,
  CONFIG: CONFIG
};

// 初始化
console.log('[CellMind] 时效记忆闭环前端模块已加载');
console.log('  强记忆窗口: 72小时');
console.log('  情感锁定: 启用');
console.log('  新窗口自动续记忆: 启用');

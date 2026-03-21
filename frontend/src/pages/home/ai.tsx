import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
  Input,
  Button,
  Space,
  Typography,
  Avatar,
  Spin,
  Select,
  message,
  Empty,
} from 'antd';
import {
  SendOutlined,
  UserOutlined,
  RobotOutlined,
  ClearOutlined,
  CopyOutlined,
  CheckOutlined,
  MenuOutlined,
} from '@ant-design/icons';
import axios from 'axios';
import { useTranslation } from 'react-i18next';
import { CREATOR_THEME } from '@/components/creator/creatorTheme';
import { md } from '@/utils/markdown';

const T = CREATOR_THEME;

/** AI 对话页专用头像背景：白色 */
const AI_AVATAR_USER = '#ffffff';
const AI_AVATAR_BOT = '#ffffff';

declare const API_URL: string;

const API_BASE = API_URL;
const { TextArea } = Input;
const { Text, Paragraph } = Typography;

// 常量定义
const STREAM_UPDATE_THROTTLE = 16; // 流式更新节流时间（毫秒），约60fps
const MAX_MESSAGE_HISTORY = 20; // 前端保留的最大消息数

// 消息类型定义
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  model?: string;
  streaming?: boolean;
}

// 模型类型
type ModelType = 'DeepSeek-v3-2' | 'Kimi-k2-5' | 'Kimi-k2' | 'GLM-4-7' | 'Gemini-3-flash' | 'Claude-Opus-4-5';

// 获取模型显示名称
const getModelDisplayName = (model: ModelType | string): string => {
  const modelMap: Record<string, string> = {
    'DeepSeek-v3-2': 'DeepSeek V3.2',
    'Kimi-k2-5': 'Kimi K2.5',
    'Kimi-k2': 'Kimi K2',
    'GLM-4-7': 'GLM-4-7',
    'Gemini-3-flash': 'Gemini 3 Flash',
    'Claude-Opus-4-5': 'Claude Opus 4.5',
  };
  return modelMap[model] || model || 'Unknown';
};

// 获取模型对应的头像路径
const getModelAvatar = (model: ModelType | string | undefined): string => {
  if (!model) return '/avatars/assistant.png';
  
  if (model === 'DeepSeek-v3-2') {
    return '/avatars/deepseek.png';
  } else if (model === 'Kimi-k2-5' || model === 'Kimi-k2') {
    return '/avatars/kimi.png';
  } else if (model === 'GLM-4-7') {
    return '/avatars/zai.png';
  } else if (model === 'Gemini-3-flash') {
    return '/avatars/google.png';
  } else if (model === 'Claude-Opus-4-5') {
    return '/avatars/claude.png';
  }
  
  return '/avatars/assistant.png';
};

// 获取用户头像路径
const getUserAvatar = (): string => {
  return '/avatars/user.png';
};

// 消息内容组件（优化 Markdown 渲染）
const MessageContent: React.FC<{ message: Message }> = React.memo(({ message }) => {
  // 使用 markdown-it 解析
  const htmlContent = React.useMemo(() => {
    try {
      return md.render(message.content || '');
    } catch (error) {
      console.error('Markdown 渲染错误:', error);
      // 如果渲染出错，返回转义的原始内容
      return (message.content || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  }, [message.content]);
  
  return (
    <>
      <div
        className="markdown-content"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
        style={{
          lineHeight: '1.8',
          wordBreak: 'break-word',
        }}
      />
      {message.streaming && (
        <Spin size="small" style={{ marginLeft: '8px', display: 'inline-block' }} />
      )}
    </>
  );
});

const AIAssistant: React.FC = () => {
  const { t } = useTranslation();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>('DeepSeek-v3-2');
  
  // 处理模型切换，自动清空对话
  const handleModelChange = useCallback((newModel: ModelType) => {
    setSelectedModel(newModel);
    setMessages([]);
    setInputValue('');
    streamEndedRef.current.clear();
    if (streamUpdateTimerRef.current) {
      clearTimeout(streamUpdateTimerRef.current);
      streamUpdateTimerRef.current = null;
    }
    message.info(t('ai.switchedTo', { model: getModelDisplayName(newModel) }));
  }, [t]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null); // 聊天消息容器的 ref
  const inputRef = useRef<any>(null);
  const streamUpdateTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastStreamUpdateRef = useRef<number>(0);
  const streamEndedRef = useRef<Set<string>>(new Set()); // 记录已结束流式响应的消息ID

  // 自动滚动到底部（固定在聊天容器内滚动，不滚动整个页面）
  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        // 在聊天容器内滚动，而不是滚动整个页面
        messagesContainerRef.current.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: 'smooth',
        });
      } else if (messagesEndRef.current) {
        // 如果容器 ref 不存在，使用 fallback（但应该避免这种情况）
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // 发送消息后自动聚焦输入框
  useEffect(() => {
    if (!loading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [loading]);

  // 复制消息内容
  const handleCopy = useCallback(async (content: string, messageId: string) => {
    try {
      // 优先使用现代的 Clipboard API
      if (navigator.clipboard && navigator.clipboard.writeText) {
        try {
          await navigator.clipboard.writeText(content);
          setCopiedId(messageId);
          message.success(t('ai.copied'));
          setTimeout(() => setCopiedId(null), 2000);
          return;
        } catch (clipboardError) {
          console.log('Clipboard API 失败，尝试 fallback 方法:', clipboardError);
        }
      }
      
      // Fallback: 使用 document.execCommand（兼容性更好）
      const textarea = document.createElement('textarea');
      textarea.value = content;
      textarea.style.position = 'fixed';
      textarea.style.left = '-999999px';
      textarea.style.top = '-999999px';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      
      try {
        const successful = document.execCommand('copy');
        document.body.removeChild(textarea);
        
        if (successful) {
          setCopiedId(messageId);
          message.success(t('ai.copied'));
          setTimeout(() => setCopiedId(null), 2000);
        } else {
          throw new Error('execCommand 复制失败');
        }
      } catch (execError) {
        document.body.removeChild(textarea);
        throw execError;
      }
    } catch (err) {
      console.error('复制失败:', err);
      message.error(t('ai.copyFailed'));
    }
  }, [t]);

  // 清空对话
  const handleClear = useCallback(() => {
    setMessages([]);
    setInputValue('');
    message.info(t('ai.cleared'));
    if (streamUpdateTimerRef.current) {
      clearTimeout(streamUpdateTimerRef.current);
      streamUpdateTimerRef.current = null;
    }
    streamEndedRef.current.clear();
  }, [t]);

  // 使用 requestAnimationFrame 优化流式消息更新，提升流畅度
  const throttledUpdateStreamMessage = useCallback((messageId: string, content: string) => {
    // 如果流式响应已结束，不再更新
    if (streamEndedRef.current.has(messageId)) {
      return;
    }
    
    const now = Date.now();
    
    // 使用 requestAnimationFrame 来优化渲染性能
    if (now - lastStreamUpdateRef.current < STREAM_UPDATE_THROTTLE) {
      if (streamUpdateTimerRef.current) {
        clearTimeout(streamUpdateTimerRef.current);
      }
      streamUpdateTimerRef.current = setTimeout(() => {
        // 再次检查流式响应是否已结束
        if (streamEndedRef.current.has(messageId)) {
          return;
        }
        requestAnimationFrame(() => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === messageId
                ? { ...msg, content, streaming: true }
                : msg
            )
          );
          lastStreamUpdateRef.current = Date.now();
          scrollToBottom();
        });
      }, STREAM_UPDATE_THROTTLE);
    } else {
      requestAnimationFrame(() => {
        // 再次检查流式响应是否已结束
        if (streamEndedRef.current.has(messageId)) {
          return;
        }
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, content, streaming: true }
              : msg
          )
        );
        lastStreamUpdateRef.current = now;
        scrollToBottom();
      });
    }
  }, [scrollToBottom]);

  // 发送消息
  const handleSend = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => {
      const newMessages = [...prev, userMessage];
      return newMessages.slice(-MAX_MESSAGE_HISTORY);
    });
    setInputValue('');
    setLoading(true);

    const assistantMessageId = `assistant-${Date.now()}-${Math.random()}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      model: selectedModel,
      streaming: true,
    };

    setMessages((prev) => [...prev, assistantMessage]);

    try {
      // 先尝试流式请求
      try {
        const response = await fetch(`${API_BASE}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: [
              ...messages.map((msg) => ({
                role: msg.role,
                content: msg.content,
              })),
              {
                role: 'user',
                content: userMessage.content,
              },
            ],
            model: selectedModel,
            stream: true,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          throw new Error('流式请求返回 JSON，使用常规请求');
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('无法读取流式响应');
        }

        const decoder = new TextDecoder();
        let fullContent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          // 解码新数据
          const chunk = decoder.decode(value, { stream: true });
          if (chunk) {
            fullContent += chunk;
            // 使用节流更新，避免过于频繁的渲染
            throttledUpdateStreamMessage(assistantMessageId, fullContent);
          }
        }

        // 标记流式响应已结束，防止节流函数再次更新状态
        streamEndedRef.current.add(assistantMessageId);
        
        // 确保清除所有待执行的定时器
        if (streamUpdateTimerRef.current) {
          clearTimeout(streamUpdateTimerRef.current);
          streamUpdateTimerRef.current = null;
        }
        
        // 重置最后更新时间
        lastStreamUpdateRef.current = 0;

        // 直接更新最终状态，确保 streaming 和 loading 都被正确设置为 false
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: fullContent, streaming: false }
              : msg
          )
        );
        setLoading(false);
        scrollToBottom();
        return;
      } catch (streamError) {
        console.log('流式请求失败，使用常规请求:', streamError);
      }

      // 使用常规请求（非流式）
      const messagesToSend = [
        ...messages.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
        {
          role: 'user',
          content: userMessage.content,
        },
      ];

      const fallbackResponse = await axios.post(`${API_BASE}/api/chat`, {
        messages: messagesToSend,
        model: selectedModel,
        stream: false,
      });

      const responseData = fallbackResponse.data;
      let responseContent = '';
      
      if (responseData.code === 0 && responseData.content) {
        responseContent = responseData.content;
      } else if (responseData.content) {
        responseContent = responseData.content;
      } else if (responseData.message) {
        responseContent = responseData.message;
      } else {
        responseContent = '抱歉，发生了错误';
      }

      // 标记流式响应已结束（非流式响应也标记，保持一致性）
      streamEndedRef.current.add(assistantMessageId);
      
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: responseContent,
                streaming: false,
              }
            : msg
        )
      );
      setLoading(false);
      scrollToBottom();
    } catch (error: any) {
      console.error('Error sending message:', error);
      
      // 标记流式响应已结束
      streamEndedRef.current.add(assistantMessageId);
      
      if (streamUpdateTimerRef.current) {
        clearTimeout(streamUpdateTimerRef.current);
        streamUpdateTimerRef.current = null;
      }
      
      let errorMessage = '抱歉，无法连接到 AI 服务。请检查网络连接或稍后重试。';
      
      if (error.response) {
        const status = error.response.status;
        if (status === 400) {
          errorMessage = '请求格式错误，请重试';
        } else if (status === 500) {
          errorMessage = '服务器内部错误，请稍后重试';
        } else if (status >= 500) {
          errorMessage = '服务器错误，请稍后重试';
        }
      } else if (error.request) {
        errorMessage = '网络连接失败，请检查网络设置';
      }
      
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: errorMessage,
                streaming: false,
              }
            : msg
        )
      );
      setLoading(false);
      message.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 处理键盘事件
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 优化：使用 useMemo 缓存模型显示名称
  const modelDisplayName = useMemo(() => getModelDisplayName(selectedModel), [selectedModel]);

  return (
    <div
      className="ai-page"
      style={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        background: T.bgPage,
        fontFamily: T.fontFamily,
        color: T.text,
      }}
    >
      {/* 顶栏 — 与创作助手统一深色 */}
      <div
        style={{
          height: 64,
          borderBottom: `1px solid ${T.borderHeader}`,
          background: T.bgHeader,
          backdropFilter: T.headerBlur,
          WebkitBackdropFilter: T.headerBlur,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Button
            type="text"
            icon={<MenuOutlined />}
            onClick={() => setSidebarVisible(!sidebarVisible)}
            style={{ fontSize: 16, color: T.textMuted }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Avatar
              src={getModelAvatar(selectedModel)}
              icon={<RobotOutlined />}
              shape="circle"
              size={32}
              className="ai-model-avatar"
              style={{ background: AI_AVATAR_BOT }}
              onError={() => false}
            />
            <div>
              <Text strong style={{ fontSize: 16, color: T.textBright }}>
                {t('ai.title')}
              </Text>
              <div style={{ fontSize: 12, color: T.textMuted, lineHeight: 1.2 }}>
                {modelDisplayName}
              </div>
            </div>
          </div>
        </div>

        <Space size="middle">
          <Select
            value={selectedModel}
            onChange={handleModelChange}
            style={{ width: 160 }}
            size="middle"
            options={[
              { value: 'DeepSeek-v3-2', label: 'DeepSeek V3.2' },
              { value: 'Kimi-k2-5', label: 'Kimi K2.5' },
              { value: 'Kimi-k2', label: 'Kimi K2' },
              { value: 'GLM-4-7', label: 'GLM-4.7' },
              { value: 'Gemini-3-flash', label: 'Gemini 3 Flash' },
              { value: 'Claude-Opus-4-5', label: 'Claude Opus 4.5' },
            ]}
            className="ai-model-select"
          />
          <Button
            type="text"
            icon={<ClearOutlined />}
            onClick={handleClear}
            disabled={messages.length === 0}
            style={{ color: T.textMuted }}
          >
            {t('ai.clear')}
          </Button>
        </Space>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {sidebarVisible && (
          <div
            style={{
              width: 280,
              borderRight: `1px solid ${T.border}`,
              background: T.bgSidebar,
              padding: 16,
              overflowY: 'auto',
            }}
          >
            <div style={{ marginBottom: 16 }}>
              <Text strong style={{ fontSize: 14, color: T.textBright }}>
                对话历史
              </Text>
            </div>
            <Empty
              description={
                <span style={{ fontSize: 12, color: T.textMuted }}>
                  暂无历史对话
                </span>
              }
              style={{ marginTop: 40 }}
            />
          </div>
        )}

        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            background: T.bgCanvas,
          }}
        >
          <div
            ref={messagesContainerRef}
            style={{
              flex: 1,
              overflowY: 'auto',
              overflowX: 'hidden',
              padding: 24,
              background: 'transparent',
              position: 'relative',
            }}
          >
            {messages.length === 0 ? (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  marginTop: -100,
                }}
              >
                <Avatar
                  src={getModelAvatar(selectedModel)}
                  icon={<RobotOutlined />}
                  shape="circle"
                  size={64}
                  className="ai-model-avatar"
                  style={{ marginBottom: 24, background: AI_AVATAR_BOT }}
                  onError={() => false}
                />
                <div style={{ fontSize: 20, fontWeight: 500, color: T.textBright, marginBottom: 8 }}>
                  开始对话
                </div>
                <div
                  style={{
                    fontSize: 14,
                    color: T.textMuted,
                    textAlign: 'center',
                    maxWidth: 400,
                  }}
                >
                  支持 DeepSeek V3.2、Kimi K2.5、Kimi K2、GLM-4.7、Gemini 3 Flash、Claude Opus 4.5
                </div>
              </div>
            ) : (
              <div style={{ maxWidth: 768, margin: '0 auto' }}>
                {messages.map((msg, index) => (
                  <div
                    key={msg.id}
                    style={{
                      marginBottom: index === messages.length - 1 ? 0 : 32,
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        gap: 16,
                        alignItems: 'flex-start',
                        flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                        justifyContent: 'flex-start',
                      }}
                    >
                      <Avatar
                        src={
                          msg.role === 'user'
                            ? getUserAvatar()
                            : getModelAvatar(msg.model)
                        }
                        icon={
                          msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />
                        }
                        shape="circle"
                        size={32}
                        style={{
                          background: msg.role === 'user' ? AI_AVATAR_USER : AI_AVATAR_BOT,
                          flexShrink: 0,
                        }}
                        className={msg.role === 'assistant' ? 'ai-model-avatar' : ''}
                        onError={() => false}
                      />
                      <div
                        style={{
                          maxWidth: msg.role === 'user' ? '70%' : '100%',
                          minWidth: msg.role === 'user' ? 120 : 'auto',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        }}
                      >
                        <div
                          style={{
                            padding: '16px 20px',
                            borderRadius: T.radiusLg,
                            background: msg.role === 'user' ? T.bgMsgUser : T.bgMsgBot,
                            border: `1px solid ${msg.role === 'user' ? T.borderMsgUser : T.borderMsgBot}`,
                            lineHeight: 1.8,
                            wordBreak: 'break-word',
                            display: 'inline-block',
                            maxWidth: '100%',
                            boxShadow: T.shadowCard,
                          }}
                        >
                          {msg.role === 'user' ? (
                            <Paragraph
                              style={{
                                margin: 0,
                                whiteSpace: 'pre-wrap',
                                color: T.textMsgUser,
                                fontSize: 16,
                                lineHeight: 1.8,
                              }}
                            >
                              {msg.content}
                            </Paragraph>
                          ) : (
                            <MessageContent message={msg} />
                          )}
                        </div>
                        {msg.role === 'assistant' && msg.content && (
                          <div
                            style={{
                              marginTop: 8,
                              display: 'flex',
                              alignItems: 'center',
                              gap: 12,
                              alignSelf: 'flex-start',
                            }}
                          >
                            <Button
                              type="text"
                              size="small"
                              icon={
                                copiedId === msg.id ? (
                                  <CheckOutlined />
                                ) : (
                                  <CopyOutlined />
                                )
                              }
                              onClick={() => handleCopy(msg.content, msg.id)}
                              style={{
                                color: T.textMuted,
                                fontSize: 12,
                                height: 24,
                                padding: '0 8px',
                              }}
                            >
                              {copiedId === msg.id ? t('ai.copiedShort') : t('ai.copy')}
                            </Button>
                            {msg.model && (
                              <span style={{ fontSize: 11, color: T.textDim }}>
                                {getModelDisplayName(msg.model)}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <div
            style={{
              borderTop: `1px solid ${T.border}`,
              background: T.bgInput,
              padding: '16px 24px',
            }}
          >
            <div
              style={{
                maxWidth: 768,
                margin: '0 auto',
                display: 'flex',
                gap: 12,
                alignItems: 'flex-end',
              }}
            >
              <div style={{ flex: 1 }}>
                <TextArea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={t('ai.placeholder')}
                  autoSize={{ minRows: 1, maxRows: 6 }}
                  disabled={loading}
                  style={{
                    background: 'rgba(24, 24, 32, 0.8)',
                    border: `1px solid ${T.borderStrong}`,
                    borderRadius: T.radiusMd,
                    padding: '12px 16px',
                    resize: 'none',
                    fontSize: 14,
                    lineHeight: 1.6,
                    color: T.text,
                  }}
                />
                <div
                  style={{
                    marginTop: 8,
                    fontSize: 12,
                    color: T.textDim,
                    textAlign: 'right',
                  }}
                >
                  {loading ? t('ai.thinking') : t('ai.footerHint')}
                </div>
              </div>
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleSend}
                loading={loading}
                disabled={!inputValue.trim()}
                size="large"
                style={{
                  height: 40,
                  width: 40,
                  padding: 0,
                  borderRadius: T.radiusSm,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: T.primaryBg,
                  borderColor: T.primaryBg,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .ai-page .ant-avatar.ai-model-avatar > img {
          object-fit: cover;
          width: 100%;
          height: 100%;
          border-radius: 50%;
        }
        .ai-page .ai-model-select.ant-select .ant-select-selector {
          background: rgba(255,255,255,0.06) !important;
          border-color: rgba(255,255,255,0.12) !important;
          color: #e4e4e7 !important;
        }
        .ai-page .ai-model-select.ant-select:hover .ant-select-selector {
          border-color: rgba(255,75,47,0.4) !important;
        }

        .ai-page .markdown-content {
          color: rgba(255,255,255,0.9);
          font-size: 16px;
        }
        .ai-page .markdown-content h1,
        .ai-page .markdown-content h2,
        .ai-page .markdown-content h3,
        .ai-page .markdown-content h4 {
          margin-top: 20px;
          margin-bottom: 12px;
          font-weight: 600;
          color: #fafafa;
        }
        .ai-page .markdown-content h1 { font-size: 1.5em; }
        .ai-page .markdown-content h2 { font-size: 1.3em; }
        .ai-page .markdown-content h3 { font-size: 1.1em; }
        .ai-page .markdown-content p {
          margin: 12px 0;
          line-height: 1.8;
        }
        .ai-page .markdown-content code {
          background: rgba(255,255,255,0.08);
          padding: 2px 6px;
          border-radius: 4px;
          font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
          font-size: 0.9em;
          color: #e4e4e7;
        }
        .ai-page .markdown-content pre {
          background: rgba(0,0,0,0.25);
          padding: 16px;
          border-radius: 8px;
          overflow-x: auto;
          margin: 16px 0;
          border: 1px solid rgba(255,255,255,0.08);
        }
        .ai-page .markdown-content pre code {
          background: none;
          padding: 0;
        }
        .ai-page .markdown-content ul,
        .ai-page .markdown-content ol {
          margin: 12px 0;
          padding-left: 24px;
        }
        .ai-page .markdown-content li { margin: 6px 0; line-height: 1.8; }
        .ai-page .markdown-content a {
          color: #ffa7a7;
          text-decoration: none;
        }
        .ai-page .markdown-content a:hover { text-decoration: underline; }
        .ai-page .markdown-content blockquote {
          border-left: 3px solid rgba(255,75,47,0.4);
          padding-left: 16px;
          margin: 16px 0;
          color: rgba(255,255,255,0.5);
          font-style: italic;
        }
        .ai-page .markdown-content table {
          border-collapse: collapse;
          width: 100%;
          margin: 16px 0;
          font-size: 0.9em;
        }
        .ai-page .markdown-content th,
        .ai-page .markdown-content td {
          border: 1px solid rgba(255,255,255,0.1);
          padding: 10px 12px;
          text-align: left;
        }
        .ai-page .markdown-content th {
          background: rgba(255,255,255,0.06);
          font-weight: 600;
        }
        .ai-page .markdown-content hr {
          border: none;
          border-top: 1px solid rgba(255,255,255,0.1);
          margin: 24px 0;
        }
      `}</style>
    </div>
  );
};

export default AIAssistant;

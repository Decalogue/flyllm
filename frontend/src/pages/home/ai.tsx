import React from 'react';
import { Button, Typography } from 'antd';
import { MessageOutlined, HomeOutlined } from '@ant-design/icons';
import { Link } from 'umi';
import { useTranslation } from 'react-i18next';
import { INTRO_THEME } from '@/components/creator/creatorTheme';
import { motion } from 'framer-motion';

const T = INTRO_THEME;
const { Title, Paragraph } = Typography;

/**
 * 项目核心 AI 入口页：复杂多轮对话在 /chat，此处作为能力台与导航。
 */
export default function AiHubPage() {
  const { t } = useTranslation();

  return (
    <div
      style={{
        fontFamily: T.fontFamily,
        color: T.text,
        background: `linear-gradient(180deg, ${T.bgPage} 0%, #fafafa 100%)`,
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        style={{
          maxWidth: 480,
          width: '100%',
          padding: 40,
          borderRadius: T.radiusLg,
          border: `1px solid ${T.border}`,
          background: '#fff',
          boxShadow: '0 12px 40px rgba(0,0,0,0.08)',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: 16,
            background: `${T.accent}18`,
            color: T.accent,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 26,
            margin: '0 auto 20px',
          }}
        >
          <MessageOutlined />
        </div>
        <Title level={3} style={{ color: T.textBright, marginBottom: 12 }}>
          {t('ai.hubPageTitle')}
        </Title>
        <Paragraph style={{ color: T.textMuted, marginBottom: 28 }}>
          {t('ai.hubPageSubtitle')}
        </Paragraph>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Link to="/chat">
            <Button type="primary" block size="large" icon={<MessageOutlined />} style={{ background: T.primaryBg, borderColor: T.primaryBg, height: 48 }}>
              {t('ai.hubPageChatCta')}
            </Button>
          </Link>
          <Link to="/">
            <Button block size="large" icon={<HomeOutlined />}>
              {t('ai.hubPageBack')}
            </Button>
          </Link>
        </div>
      </motion.div>
    </div>
  );
}

import React, { useState } from 'react';
import { Button, Typography, Alert, Space } from 'antd';
import { ArrowLeftOutlined, CheckOutlined, RollbackOutlined } from '@ant-design/icons';
import { Link } from 'umi';
import { useTranslation } from 'react-i18next';
import { INTRO_THEME } from '@/components/creator/creatorTheme';
import { motion, AnimatePresence } from 'framer-motion';

const T = INTRO_THEME;
const { Title, Paragraph, Text } = Typography;

/** 静态演示卡片，后续替换为调度队列 + 持久化 */
const MOCK_CARDS = [
  { id: '1', frontKey: 'review.card1Front', backKey: 'review.card1Back' },
  { id: '2', frontKey: 'review.card2Front', backKey: 'review.card2Back' },
];

export default function ReviewTodayPage() {
  const { t } = useTranslation();
  const [index, setIndex] = useState(0);
  const [showBack, setShowBack] = useState(false);
  const [done, setDone] = useState(false);

  const current = MOCK_CARDS[index];
  const isLast = index >= MOCK_CARDS.length - 1;

  const nextCard = () => {
    if (isLast) {
      setDone(true);
      return;
    }
    setIndex((i) => i + 1);
    setShowBack(false);
  };

  return (
    <div
      style={{
        fontFamily: T.fontFamily,
        color: T.text,
        background: T.bgPage,
        minHeight: '100vh',
        padding: '48px 24px 80px',
      }}
    >
      <div style={{ maxWidth: 640, margin: '0 auto' }}>
        <Link to="/review" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: T.textMuted, marginBottom: 24 }}>
          <ArrowLeftOutlined /> {t('review.backHub')}
        </Link>

        <Title level={2} style={{ color: T.textBright, marginBottom: 8 }}>
          {t('review.todayTitle')}
        </Title>
        <Alert type="info" showIcon message={t('review.todayMockHint')} style={{ marginBottom: 24 }} />

        {done ? (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Paragraph>{t('review.todayDone')}</Paragraph>
            <Space>
              <Link to="/review">
                <Button type="primary" style={{ background: T.primaryBg, borderColor: T.primaryBg }}>
                  {t('review.backHub')}
                </Button>
              </Link>
              <Button
                onClick={() => {
                  setIndex(0);
                  setShowBack(false);
                  setDone(false);
                }}
              >
                {t('review.todayAgain')}
              </Button>
            </Space>
          </motion.div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={current.id}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.25 }}
              style={{
                padding: 32,
                borderRadius: T.radiusLg,
                border: `1px solid ${T.border}`,
                background: '#fff',
                boxShadow: '0 8px 24px rgba(0,0,0,0.06)',
                minHeight: 220,
              }}
            >
              <Text type="secondary" style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {t('review.cardProgress', { current: index + 1, total: MOCK_CARDS.length })}
              </Text>
              <Paragraph style={{ fontSize: 18, color: T.textBright, marginTop: 12, marginBottom: showBack ? 16 : 0 }}>
                {t(current.frontKey)}
              </Paragraph>
              {showBack && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <Paragraph style={{ color: T.textMuted, fontSize: 15, lineHeight: 1.7, borderLeft: `3px solid ${T.accent}`, paddingLeft: 16 }}>
                    {t(current.backKey)}
                  </Paragraph>
                </motion.div>
              )}

              <Space wrap style={{ marginTop: 28 }}>
                {!showBack ? (
                  <Button type="primary" onClick={() => setShowBack(true)} style={{ background: T.primaryBg, borderColor: T.primaryBg }}>
                    {t('review.revealAnswer')}
                  </Button>
                ) : (
                  <>
                    <Button icon={<RollbackOutlined />} onClick={nextCard}>
                      {t('review.gradeAgain')}
                    </Button>
                    <Button type="primary" icon={<CheckOutlined />} onClick={nextCard} style={{ background: T.primaryBg, borderColor: T.primaryBg }}>
                      {t('review.gradeGood')}
                    </Button>
                  </>
                )}
              </Space>
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

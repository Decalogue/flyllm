import React from 'react';
import { Button, Card, Typography } from 'antd';
import { ArrowLeftOutlined, BookOutlined, CalendarOutlined, NodeIndexOutlined } from '@ant-design/icons';
import { Link } from 'umi';
import { useTranslation } from 'react-i18next';
import { INTRO_THEME } from '@/components/creator/creatorTheme';
import { motion } from 'framer-motion';

const T = INTRO_THEME;
const { Title, Paragraph } = Typography;

export default function ReviewHubPage() {
  const { t } = useTranslation();

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
      <div style={{ maxWidth: 880, margin: '0 auto' }}>
        <Link to="/" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: T.textMuted, marginBottom: 24 }}>
          <ArrowLeftOutlined /> {t('review.hubBackHome')}
        </Link>

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <Title level={2} style={{ color: T.textBright, marginBottom: 8 }}>
            {t('review.hubTitle')}
          </Title>
          <Paragraph style={{ color: T.textMuted, fontSize: 16, marginBottom: 40, maxWidth: 560 }}>
            {t('review.hubSubtitle')}
          </Paragraph>
        </motion.div>

        <div style={{ display: 'grid', gap: 20, gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
          <Card
            hoverable
            styles={{ body: { padding: 24 } }}
            style={{ borderRadius: T.radiusLg, border: `1px solid ${T.border}` }}
          >
            <CalendarOutlined style={{ fontSize: 28, color: T.accent, marginBottom: 12 }} />
            <Title level={4} style={{ marginTop: 0, color: T.textBright }}>
              {t('review.cardTodayTitle')}
            </Title>
            <Paragraph type="secondary" style={{ minHeight: 48 }}>
              {t('review.cardTodayDesc')}
            </Paragraph>
            <Link to="/review/today">
              <Button type="primary" style={{ background: T.primaryBg, borderColor: T.primaryBg }}>
                {t('review.cardTodayAction')}
              </Button>
            </Link>
          </Card>

          <Card
            styles={{ body: { padding: 24 } }}
            style={{ borderRadius: T.radiusLg, border: `1px dashed ${T.border}`, opacity: 0.85 }}
          >
            <NodeIndexOutlined style={{ fontSize: 28, color: T.textDim, marginBottom: 12 }} />
            <Title level={4} style={{ marginTop: 0, color: T.textBright }}>
              {t('review.cardSyllabusTitle')}
            </Title>
            <Paragraph type="secondary" style={{ minHeight: 48 }}>
              {t('review.cardSyllabusDesc')}
            </Paragraph>
            <Button disabled>{t('review.cardSyllabusAction')}</Button>
          </Card>

          <Card
            hoverable
            styles={{ body: { padding: 24 } }}
            style={{ borderRadius: T.radiusLg, border: `1px solid ${T.border}` }}
          >
            <BookOutlined style={{ fontSize: 28, color: '#3b82f6', marginBottom: 12 }} />
            <Title level={4} style={{ marginTop: 0, color: T.textBright }}>
              {t('review.cardStarsTitle')}
            </Title>
            <Paragraph type="secondary" style={{ minHeight: 48 }}>
              {t('review.cardStarsDesc')}
            </Paragraph>
            <Link to="/stars">
              <Button>{t('review.cardStarsAction')}</Button>
            </Link>
          </Card>
        </div>
      </div>
    </div>
  );
}

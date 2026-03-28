import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Button, Typography } from 'antd';
import {
  ThunderboltOutlined,
  ReadOutlined,
  NodeIndexOutlined,
  ArrowRightOutlined,
  MessageOutlined,
  StarOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  RocketOutlined,
  ScheduleOutlined,
  DownOutlined,
} from '@ant-design/icons';
import { motion, useMotionValue, useTransform, useReducedMotion } from 'framer-motion';
import { Link } from 'umi';
import { useTranslation } from 'react-i18next';
import { INTRO_THEME } from '@/components/creator/creatorTheme';
import { IntroFooter } from '@/components/creator/IntroFooter';
import { ReviewWorkflowLazy } from '@/pages/home/ReviewWorkflowLazy';
import '@/pages/home/review-home.css';

const T = INTRO_THEME;
const { Title, Paragraph, Text } = Typography;

const EASE = [0.25, 0.46, 0.45, 0.94] as const;

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.032, delayChildren: 0.15 } },
};
const charItem = { hidden: { y: '1em', opacity: 0 }, visible: { y: 0, opacity: 1 } };

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const fn = () => setMatches(mq.matches);
    fn();
    mq.addEventListener('change', fn);
    return () => mq.removeEventListener('change', fn);
  }, [query]);
  return matches;
}

/** 主按钮、次按钮、区块标题等复用样式 */
function useLandingStyles() {
  return useMemo(
    () => ({
      primaryBtn: {
        background: T.primaryBg,
        borderColor: T.primaryBg,
        height: 52,
        paddingLeft: 28,
        paddingRight: 28,
        fontSize: 16,
        fontWeight: T.fontWeightSemibold,
        borderRadius: T.radiusMd,
        boxShadow: '0 4px 14px rgba(255,75,47,0.35)',
      } as const,
      secondaryBtn: {
        height: 52,
        paddingLeft: 28,
        paddingRight: 28,
        fontSize: 16,
        color: T.text,
        border: `2px solid ${T.stroke}`,
        background: 'transparent',
        fontWeight: T.fontWeightMedium,
        borderRadius: T.radiusMd,
      } as const,
      sectionTitle: {
        color: T.textBright,
        fontWeight: T.fontWeightBold,
        marginBottom: 12,
        fontSize: 'clamp(24px, 4vw, 32px)',
      } as const,
    }),
    []
  );
}

export default function HomePage() {
  const { t } = useTranslation();
  const prefersReducedMotion = useReducedMotion();
  const isWideSteps = useMediaQuery('(min-width: 900px)');
  const styles = useLandingStyles();

  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const [mobileStep, setMobileStep] = useState(0);
  const heroRef = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0.5);
  const mouseY = useMotionValue(0.5);

  const [graphSize, setGraphSize] = useState({ w: 950, h: 480 });
  useEffect(() => {
    let raf = 0;
    const measure = () => {
      const vw = window.innerWidth;
      const maxW = Math.min(950, Math.max(280, vw - 32));
      const h = Math.max(300, Math.round(480 * (maxW / 950)));
      setGraphSize({ w: maxW, h });
    };
    const onResize = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener('resize', onResize, { passive: true });
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, []);

  const workflowSteps = useMemo(
    () => [
      {
        id: 'syllabus',
        title: t('reviewLanding.step1Title'),
        subtitle: t('reviewLanding.step1Subtitle'),
        description: t('reviewLanding.step1Desc'),
        icon: <ReadOutlined />,
        color: '#3b82f6',
        features: t('reviewLanding.step1Features', { returnObjects: true }) as string[],
      },
      {
        id: 'spacing',
        title: t('reviewLanding.step2Title'),
        subtitle: t('reviewLanding.step2Subtitle'),
        description: t('reviewLanding.step2Desc'),
        icon: <ScheduleOutlined />,
        color: '#10b981',
        features: t('reviewLanding.step2Features', { returnObjects: true }) as string[],
      },
      {
        id: 'starmap',
        title: t('reviewLanding.step3Title'),
        subtitle: t('reviewLanding.step3Subtitle'),
        description: t('reviewLanding.step3Desc'),
        icon: <StarOutlined />,
        color: '#f59e0b',
        features: t('reviewLanding.step3Features', { returnObjects: true }) as string[],
      },
    ],
    [t]
  );

  const titleLine1 = t('reviewLanding.titleLine1');
  const titleLine2 = t('reviewLanding.titleLine2');

  const stepActive = useCallback(
    (index: number) => {
      if (isWideSteps) return hoveredStep === index;
      return mobileStep === index;
    },
    [isWideSteps, hoveredStep, mobileStep]
  );

  const stepCollapsed = useCallback(
    (index: number) => {
      if (isWideSteps) return hoveredStep !== null && hoveredStep !== index;
      return !stepActive(index);
    },
    [isWideSteps, hoveredStep, stepActive]
  );

  const stepFlex = useCallback(
    (index: number) => {
      if (isWideSteps) {
        if (hoveredStep === null) return 1;
        return hoveredStep === index ? 2 : 0.6;
      }
      return stepActive(index) ? 2 : 0.65;
    },
    [isWideSteps, hoveredStep, stepActive]
  );

  useEffect(() => {
    const el = heroRef.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      const rw = rect.width || 1;
      const rh = rect.height || 1;
      mouseX.set((e.clientX - rect.left) / rw);
      mouseY.set((e.clientY - rect.top) / rh);
    };
    el.addEventListener('mousemove', onMove, { passive: true });
    return () => el.removeEventListener('mousemove', onMove);
  }, [mouseX, mouseY]);

  const orb1X = useTransform(mouseX, [0, 1], ['2%', '8%']);
  const orb1Y = useTransform(mouseY, [0, 1], ['10%', '20%']);
  const orb2X = useTransform(mouseX, [0, 1], ['88%', '94%']);
  const orb2Y = useTransform(mouseY, [0, 1], ['60%', '75%']);

  const metricItems = useMemo(
    () => [
      { icon: <ApiOutlined />, value: t('reviewLanding.metricRecall'), label: t('reviewLanding.metricRecallLabel'), color: T.accent },
      { icon: <ScheduleOutlined />, value: t('reviewLanding.metricSchedule'), label: t('reviewLanding.metricScheduleLabel'), color: '#10b981' },
      { icon: <CheckCircleOutlined />, value: t('reviewLanding.metricChain'), label: t('reviewLanding.metricChainLabel'), color: '#3b82f6' },
      { icon: <RocketOutlined />, value: t('reviewLanding.metricVersion'), label: t('reviewLanding.metricVersionLabel'), color: '#f59e0b' },
    ],
    [t]
  );

  const statItems = useMemo(
    () => [
      { value: t('reviewLanding.stat1'), label: t('reviewLanding.stat1Label') },
      { value: t('reviewLanding.stat2'), label: t('reviewLanding.stat2Label') },
      { value: t('reviewLanding.stat3'), label: t('reviewLanding.stat3Label') },
    ],
    [t]
  );

  const anchorLinkStyle: React.CSSProperties = {
    color: T.textMuted,
    fontSize: 14,
    textDecoration: 'none',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
  };

  return (
    <div
      className="review-home"
      style={
        {
          fontFamily: T.fontFamily,
          color: T.text,
          background: T.bgPage,
          minHeight: '100%',
          ['--review-radius-lg' as string]: `${T.radiusLg}px`,
          ['--review-border' as string]: T.border,
        } as React.CSSProperties
      }
    >
      <a href="#review-flow" className="review-skip-link">
        {t('reviewLanding.skipToContent')}
      </a>

      <main id="review-landing-main">
        <section
          ref={heroRef}
          aria-label={t('layout.navHome')}
          style={{
            minHeight: 'min(100vh, 920px)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 'clamp(48px, 10vw, 80px) clamp(20px, 4vw, 40px) clamp(64px, 12vw, 100px)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div className="review-gradient-bg" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} aria-hidden />
          <div className="review-noise" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} aria-hidden />
          {!prefersReducedMotion && (
            <>
              <motion.div
                className="review-float"
                style={{
                  position: 'absolute',
                  width: 320,
                  height: 320,
                  borderRadius: '50%',
                  background: 'radial-gradient(circle, rgba(255,75,47,0.08) 0%, transparent 70%)',
                  filter: 'blur(40px)',
                  pointerEvents: 'none',
                  x: orb1X,
                  y: orb1Y,
                  left: 0,
                  top: 0,
                }}
                aria-hidden
              />
              <motion.div
                className="review-float-delay"
                style={{
                  position: 'absolute',
                  width: 280,
                  height: 280,
                  borderRadius: '50%',
                  background: 'radial-gradient(circle, rgba(255,167,167,0.1) 0%, transparent 70%)',
                  filter: 'blur(36px)',
                  pointerEvents: 'none',
                  x: orb2X,
                  y: orb2Y,
                  right: 0,
                  bottom: 0,
                }}
                aria-hidden
              />
            </>
          )}

          <motion.div
            initial={prefersReducedMotion ? false : { opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: prefersReducedMotion ? 0 : 0.6, ease: EASE }}
            style={{ maxWidth: 960, textAlign: 'center', position: 'relative', zIndex: 1 }}
          >
            <motion.div
              initial={prefersReducedMotion ? false : { opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: prefersReducedMotion ? 0 : 0.1, duration: prefersReducedMotion ? 0 : 0.5, type: 'spring', stiffness: 200 }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 16px',
                borderRadius: 999,
                border: `1px solid ${T.accentBorder}`,
                background: T.bgBadge,
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                marginBottom: 28,
                color: T.accent,
                fontSize: 15,
                fontWeight: T.fontWeightMedium,
                boxShadow: T.shadowCard,
              }}
            >
              <span style={{ position: 'relative', display: 'inline-flex', width: 14, height: 14, alignItems: 'center', justifyContent: 'center' }}>
                <span className="review-ping" style={{ position: 'absolute', width: 8, height: 8, borderRadius: '50%', background: T.accent, opacity: 0.75 }} />
                <span style={{ position: 'relative', width: 6, height: 6, borderRadius: '50%', background: T.accent }} />
              </span>
              <ThunderboltOutlined style={{ fontSize: 15 }} aria-hidden />
              {t('reviewLanding.badge')}
            </motion.div>

            {prefersReducedMotion ? (
              <h1
                style={{
                  color: T.textBright,
                  fontWeight: T.fontWeightBold,
                  fontSize: 'clamp(32px, 5.5vw, 56px)',
                  lineHeight: 1.25,
                  margin: '0 auto 16px',
                }}
              >
                {titleLine1}
                <br />
                <span style={{ color: T.accent }}>{titleLine2}</span>
              </h1>
            ) : (
              <motion.h1
                variants={container}
                initial="hidden"
                animate="visible"
                style={{
                  color: T.textBright,
                  fontWeight: T.fontWeightBold,
                  fontSize: 'clamp(32px, 5.5vw, 56px)',
                  lineHeight: 1.25,
                  margin: '0 auto 16px',
                }}
              >
                {titleLine1.split('').map((c, i) => (
                  <motion.span key={`1-${i}`} className="review-char" variants={charItem}>
                    <span className="review-char-inner">{c === ' ' ? '\u00A0' : c}</span>
                  </motion.span>
                ))}
                <br />
                {titleLine2.split('').map((c, i) => (
                  <motion.span key={`2-${i}`} className="review-char" variants={charItem} style={{ color: T.accent }}>
                    <span className="review-char-inner">{c === ' ' ? '\u00A0' : c}</span>
                  </motion.span>
                ))}
              </motion.h1>
            )}

            <motion.p
              initial={prefersReducedMotion ? false : { opacity: 0, filter: 'blur(10px)' }}
              animate={{ opacity: 1, filter: 'blur(0px)' }}
              transition={{ duration: prefersReducedMotion ? 0 : 0.9, delay: prefersReducedMotion ? 0 : 0.5, ease: EASE }}
              style={{
                color: T.textMuted,
                fontSize: 'clamp(16px, 2.2vw, 19px)',
                lineHeight: 1.7,
                maxWidth: 672,
                margin: '0 auto 36px',
                letterSpacing: '0.02em',
              }}
            >
              {t('reviewLanding.subtitle')}
            </motion.p>

            <motion.div
              initial="hidden"
              animate="visible"
              variants={{ hidden: {}, visible: { transition: { staggerChildren: prefersReducedMotion ? 0 : 0.08, delayChildren: prefersReducedMotion ? 0 : 0.6 } } }}
              style={{ display: 'flex', flexWrap: 'wrap', gap: 18, justifyContent: 'center' }}
            >
              <motion.div
                variants={{ hidden: { scale: 0, opacity: 0 }, visible: { scale: 1, opacity: 1 } }}
                transition={{ type: 'spring', stiffness: 260, damping: 18 }}
                whileHover={prefersReducedMotion ? undefined : { scale: 1.03 }}
                whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}
              >
                <Link to="/review/today">
                  <Button type="primary" size="large" icon={<ReadOutlined />} style={styles.primaryBtn}>
                    {t('reviewLanding.ctaReview')}
                  </Button>
                </Link>
              </motion.div>
              <motion.div
                variants={{ hidden: { scale: 0, opacity: 0 }, visible: { scale: 1, opacity: 1 } }}
                transition={{ type: 'spring', stiffness: 260, damping: 18 }}
                whileHover={prefersReducedMotion ? undefined : { scale: 1.03 }}
                whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}
              >
                <Link to="/ai">
                  <Button size="large" icon={<MessageOutlined />} style={styles.secondaryBtn}>
                    {t('reviewLanding.ctaAI')}
                  </Button>
                </Link>
              </motion.div>
            </motion.div>

            <nav aria-label={t('reviewLanding.navOnPage')} style={{ marginTop: 20, marginBottom: 8 }}>
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '12px 20px',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <a href="#review-flow" style={anchorLinkStyle}>
                  <DownOutlined style={{ fontSize: 12, opacity: 0.65 }} aria-hidden />
                  {t('reviewLanding.anchorFlow')}
                </a>
                <span style={{ color: T.border, userSelect: 'none' }} aria-hidden>
                  ·
                </span>
                <a href="#review-steps" style={anchorLinkStyle}>
                  {t('reviewLanding.anchorSteps')}
                </a>
                <span style={{ color: T.border, userSelect: 'none' }} aria-hidden>
                  ·
                </span>
                <a href="#review-starmap" style={anchorLinkStyle}>
                  {t('reviewLanding.anchorStars')}
                </a>
              </div>
            </nav>

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: prefersReducedMotion ? 0 : 0.6, delay: prefersReducedMotion ? 0 : 0.85 }}
              style={{ display: 'flex', flexWrap: 'wrap', gap: 20, justifyContent: 'center', marginTop: 12 }}
            >
              <Link to="/chat" style={{ color: T.textMuted, fontSize: 15 }}>
                {t('reviewLanding.ctaChat')}
              </Link>
              <span style={{ color: T.border }} aria-hidden>
                |
              </span>
              <Link to="/stars" style={{ color: T.textMuted, fontSize: 15 }}>
                {t('reviewLanding.ctaStars')}
              </Link>
              <span style={{ color: T.border }} aria-hidden>
                |
              </span>
              <Link to="/review" style={{ color: T.textMuted, fontSize: 15 }}>
                {t('reviewLanding.ctaHub')}
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: prefersReducedMotion ? 0 : 0.6, delay: prefersReducedMotion ? 0 : 0.95 }}
              style={{ display: 'flex', flexWrap: 'wrap', gap: 40, justifyContent: 'center', marginTop: 40 }}
              role="list"
            >
              {statItems.map((s, i) => (
                <motion.div
                  key={i}
                  style={{ textAlign: 'center' }}
                  role="listitem"
                  whileHover={prefersReducedMotion ? undefined : { y: -2 }}
                  transition={{ duration: 0.2 }}
                >
                  <div style={{ fontSize: 18, fontWeight: T.fontWeightBold, color: T.textBright }}>{s.value}</div>
                  <div style={{ fontSize: 13, color: T.textDim }}>{s.label}</div>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        </section>

        <section
          id="review-flow"
          className="review-section-anchor"
          aria-labelledby="review-flow-heading"
          style={{
            padding: 'clamp(72px, 12vw, 120px) clamp(20px, 4vw, 40px) clamp(80px, 14vw, 140px)',
            background: 'linear-gradient(180deg, #fafafa 0%, #ffffff 100%)',
            minHeight: 'min(100vh, 900px)',
          }}
        >
          <div style={{ maxWidth: 1280, margin: '0 auto' }}>
            <motion.div
              initial={{ opacity: 0, y: 32 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.6, ease: EASE }}
              style={{ textAlign: 'center', marginBottom: 48 }}
            >
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 14px',
                  borderRadius: 999,
                  background: 'rgba(255,75,47,0.1)',
                  border: `1px solid ${T.accentBorder}`,
                  marginBottom: 16,
                }}
              >
                <NodeIndexOutlined style={{ color: T.accent, fontSize: 14 }} aria-hidden />
                <span style={{ fontSize: 14, fontWeight: T.fontWeightSemibold, color: T.accent }}>{t('reviewLanding.flowBadge')}</span>
              </div>
              <Title id="review-flow-heading" level={2} style={{ ...styles.sectionTitle, marginBottom: 12 }}>
                {t('reviewLanding.flowTitle')} <span className="review-text-gradient">{t('reviewLanding.flowTitleHighlight')}</span>
              </Title>
              <Paragraph style={{ color: T.textMuted, fontSize: 18, maxWidth: 672, margin: '0 auto' }}>
                {t('reviewLanding.flowDesc')}
              </Paragraph>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40, scale: 0.98 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ duration: 0.7, ease: EASE }}
              style={{ marginBottom: 40, overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}
            >
              <div style={{ minWidth: Math.min(graphSize.w, 950), margin: '0 auto' }}>
                <ReviewWorkflowLazy
                  variant="research"
                  demo
                  showAgentCenter
                  scale={1}
                  height={graphSize.h}
                  width={graphSize.w}
                  loadingLabel={t('reviewLanding.workflowLoading')}
                />
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.15 }}
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 160px), 1fr))',
                gap: 20,
                marginBottom: 40,
              }}
            >
              {metricItems.map((s, i) => (
                <div key={i} className="review-metric-card">
                  <div
                    style={{
                      width: 52,
                      height: 52,
                      borderRadius: 12,
                      background: `${s.color}18`,
                      color: s.color,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 22,
                      margin: '0 auto 14px',
                    }}
                  >
                    {s.icon}
                  </div>
                  <div style={{ fontSize: 20, fontWeight: T.fontWeightBold, color: T.textBright, marginBottom: 4 }}>{s.value}</div>
                  <div style={{ fontSize: 14, color: T.textDim }}>{s.label}</div>
                </div>
              ))}
            </motion.div>

            <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} style={{ textAlign: 'center' }}>
              <Link to="/review">
                <Button type="primary" size="large" icon={<ArrowRightOutlined />} style={{ ...styles.primaryBtn, paddingLeft: 32, paddingRight: 32 }}>
                  {t('reviewLanding.flowCta')}
                </Button>
              </Link>
            </motion.div>
          </div>
        </section>

        <section
          id="review-steps"
          className="review-section-anchor"
          aria-labelledby="review-steps-heading"
          style={{
            padding: 'clamp(64px, 10vw, 100px) clamp(20px, 4vw, 40px) clamp(72px, 12vw, 120px)',
            maxWidth: 1280,
            margin: '0 auto',
            minHeight: 'min(100vh, 820px)',
            background: T.bgPage,
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.6, ease: EASE }}
            style={{ textAlign: 'center', marginBottom: 48 }}
          >
            <Title id="review-steps-heading" level={2} style={{ ...styles.sectionTitle, marginBottom: 8 }}>
              {t('reviewLanding.stepsTitle')} <span className="review-text-gradient">{t('reviewLanding.stepsTitleHighlight')}</span>
            </Title>
            <Text style={{ color: T.textMuted, fontSize: 18 }}>{t('reviewLanding.stepsSubtitle')}</Text>
            {!isWideSteps && (
              <Text type="secondary" style={{ display: 'block', marginTop: 12, fontSize: 13 }}>
                {t('reviewLanding.tapToSwitchStep')}
              </Text>
            )}
          </motion.div>

          <motion.div
            layout
            aria-label={t('reviewLanding.sectionStepsAria')}
            style={{ display: 'flex', gap: 16, minHeight: 420 }}
            className="review-steps-container"
            onMouseLeave={() => isWideSteps && setHoveredStep(null)}
          >
            {workflowSteps.map((step, index) => {
              const active = stepActive(index);
              const collapsed = stepCollapsed(index);
              return (
                <motion.div
                  key={step.id}
                  layout
                  tabIndex={isWideSteps ? undefined : 0}
                  initial={{ opacity: 0, y: 36 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-40px' }}
                  transition={{ delay: index * 0.1, duration: 0.5, ease: EASE }}
                  onMouseEnter={() => isWideSteps && setHoveredStep(index)}
                  onClick={() => {
                    if (isWideSteps) setHoveredStep(index);
                    else setMobileStep(index);
                  }}
                  onKeyDown={(e) => {
                    if (!isWideSteps && (e.key === 'Enter' || e.key === ' ')) {
                      e.preventDefault();
                      setMobileStep(index);
                    }
                  }}
                  style={{
                    flex: stepFlex(index),
                    minHeight: active ? 340 : collapsed ? 120 : 220,
                    display: 'flex',
                    flexDirection: 'column',
                    position: 'relative',
                    borderRadius: T.radiusLg,
                    overflow: 'hidden',
                    cursor: 'pointer',
                    transition: 'flex 0.45s cubic-bezier(0.25, 0.46, 0.45, 0.94), min-height 0.45s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                  }}
                  className="review-step-card"
                >
                  <motion.div
                    layout
                    transition={{ duration: 0.35 }}
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background: `linear-gradient(135deg, ${step.color}18 0%, ${step.color}08 100%)`,
                      opacity: active ? 1 : 0,
                    }}
                  />
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'rgba(248,248,248,0.95)',
                      opacity: active ? 0 : 1,
                      transition: 'opacity 0.35s',
                    }}
                  />
                  <motion.div
                    layout
                    transition={{ duration: 0.25 }}
                    style={{
                      position: 'absolute',
                      inset: 0,
                      borderRadius: T.radiusLg,
                      border: `2px solid ${step.color}`,
                      opacity: active ? 1 : 0,
                      pointerEvents: 'none',
                    }}
                  />

                  <div style={{ position: 'relative', flex: 1, padding: 20, display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
                      <motion.div
                        animate={active ? { scale: 1.06 } : { scale: 1 }}
                        transition={{ type: 'spring', stiffness: 300, damping: 22 }}
                        style={{
                          width: 52,
                          height: 52,
                          borderRadius: T.radiusMd,
                          background: active ? step.color : `${step.color}18`,
                          color: active ? '#fff' : step.color,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 24,
                          flexShrink: 0,
                        }}
                      >
                        {step.icon}
                      </motion.div>
                      <span
                        style={{
                          fontSize: 44,
                          fontWeight: T.fontWeightBold,
                          color: step.color,
                          lineHeight: 1,
                          opacity: active ? 0.28 : 0.12,
                        }}
                        aria-hidden
                      >
                        0{index + 1}
                      </span>
                    </div>

                    <div style={{ marginTop: 10, marginBottom: 6 }}>
                      <Text style={{ color: T.textDim, fontSize: 12, fontWeight: T.fontWeightMedium, display: 'block', marginBottom: 4 }}>{step.subtitle}</Text>
                      <Title level={4} style={{ color: T.textBright, margin: 0, fontWeight: T.fontWeightSemibold, fontSize: 22 }}>
                        {step.title}
                      </Title>
                    </div>

                    <motion.div
                      animate={{
                        opacity: active ? 1 : 0,
                        maxHeight: active ? 320 : 0,
                      }}
                      transition={{ duration: 0.3, ease: EASE }}
                      style={{ overflow: 'hidden', flex: 1 }}
                      aria-hidden={!active}
                    >
                      <Paragraph style={{ color: T.textMuted, fontSize: 15, lineHeight: 1.65, marginBottom: 12 }}>
                        {step.description}
                      </Paragraph>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {step.features.map((f, i) => (
                          <span
                            key={i}
                            style={{
                              padding: '5px 12px',
                              borderRadius: 999,
                              fontSize: 13,
                              fontWeight: T.fontWeightMedium,
                              background: `${step.color}15`,
                              color: step.color,
                            }}
                          >
                            {f}
                          </span>
                        ))}
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </section>

        <section
          id="review-starmap"
          className="review-section-anchor"
          aria-labelledby="review-starmap-heading"
          style={{
            padding: 'clamp(72px, 12vw, 120px) clamp(20px, 4vw, 40px) clamp(80px, 14vw, 140px)',
            background: 'linear-gradient(180deg, #0a0a0a 0%, #0f0f1a 40%, #1a1a2e 100%)',
            position: 'relative',
            overflow: 'hidden',
            minHeight: 'min(100vh, 640px)',
          }}
        >
          <div style={{ position: 'absolute', top: '20%', left: '15%', width: 280, height: 280, borderRadius: '50%', background: 'rgba(255,75,47,0.06)', filter: 'blur(60px)', pointerEvents: 'none' }} aria-hidden />
          <div style={{ position: 'absolute', bottom: '20%', right: '15%', width: 240, height: 240, borderRadius: '50%', background: 'rgba(59,130,246,0.06)', filter: 'blur(50px)', pointerEvents: 'none' }} aria-hidden />
          <div
            style={{
              position: 'absolute',
              inset: 0,
              opacity: 0.08,
              pointerEvents: 'none',
              backgroundImage: 'linear-gradient(to right, #ff4b2f 1px, transparent 1px), linear-gradient(to bottom, #ff4b2f 1px, transparent 1px)',
              backgroundSize: '48px 48px',
            }}
            aria-hidden
          />

          <div style={{ maxWidth: 720, margin: '0 auto', position: 'relative', zIndex: 1, textAlign: 'center' }}>
            <motion.div
              initial={{ opacity: 0, y: 32 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.6, ease: EASE }}
            >
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 14px',
                  borderRadius: 999,
                  background: 'rgba(255,75,47,0.15)',
                  border: '1px solid rgba(255,75,47,0.3)',
                  marginBottom: 16,
                }}
              >
                <StarOutlined style={{ color: T.accent, fontSize: 14 }} aria-hidden />
                <span style={{ fontSize: 13, fontWeight: T.fontWeightBold, color: T.accent, letterSpacing: '0.04em' }}>{t('reviewLanding.starsBadge')}</span>
              </div>
              <Title id="review-starmap-heading" level={2} style={{ color: '#fafafa', fontWeight: T.fontWeightBold, marginBottom: 12, fontSize: 'clamp(24px, 4vw, 32px)' }}>
                {t('reviewLanding.starsTitle')} <span className="review-text-gradient">{t('reviewLanding.starsTitleHighlight')}</span>
              </Title>
              <Paragraph style={{ color: 'rgba(255,255,255,0.55)', fontSize: 18, maxWidth: 560, margin: '0 auto 32px' }}>
                {t('reviewLanding.starsDesc')}
              </Paragraph>
              <Link to="/stars">
                <Button type="primary" size="large" icon={<StarOutlined />} style={{ ...styles.primaryBtn, paddingLeft: 32, paddingRight: 32 }}>
                  {t('reviewLanding.starsCta')}
                </Button>
              </Link>
            </motion.div>
          </div>
        </section>

        <section
          aria-labelledby="review-cta-heading"
          style={{
            padding: 'clamp(64px, 10vw, 100px) clamp(20px, 4vw, 40px) clamp(72px, 12vw, 120px)',
            textAlign: 'center',
            minHeight: 'min(80vh, 520px)',
            background: 'linear-gradient(180deg, #ffffff 0%, #fafafa 100%)',
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: EASE }}
            style={{
              maxWidth: 720,
              margin: '0 auto',
              padding: 'clamp(40px, 8vw, 64px) clamp(24px, 5vw, 48px)',
              borderRadius: T.radiusXl,
              background: T.accentDim,
              border: `1px solid ${T.accentBorder}`,
              boxShadow: T.shadowCta,
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <div className="review-shimmer" style={{ position: 'absolute', inset: 0, pointerEvents: 'none', borderRadius: 'inherit' }} aria-hidden />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <Title id="review-cta-heading" level={3} style={{ color: T.textBright, marginBottom: 12, fontWeight: T.fontWeightBold, fontSize: 'clamp(22px, 3.5vw, 28px)' }}>
                {t('reviewLanding.finalTitle')}
              </Title>
              <Paragraph style={{ color: T.textMuted, marginBottom: 24, fontSize: 17, maxWidth: 560, margin: '0 auto 24px' }}>
                {t('reviewLanding.finalSubtitle')}
              </Paragraph>
              <motion.div whileHover={prefersReducedMotion ? undefined : { scale: 1.03 }} whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}>
                <Link to="/review/today">
                  <Button type="primary" size="large" icon={<ArrowRightOutlined />} style={{ ...styles.primaryBtn, paddingLeft: 32, paddingRight: 32 }}>
                    {t('reviewLanding.finalCta')}
                  </Button>
                </Link>
              </motion.div>
            </div>
          </motion.div>
        </section>
      </main>

      <IntroFooter />
    </div>
  );
}

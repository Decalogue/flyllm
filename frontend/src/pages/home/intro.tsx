import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Button, Typography } from 'antd';
import {
  ThunderboltOutlined,
  BulbOutlined,
  EditOutlined,
  RiseOutlined,
  ArrowRightOutlined,
  MessageOutlined,
  DatabaseOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  RocketOutlined,
} from '@ant-design/icons';
import { motion, useMotionValue, useTransform } from 'framer-motion';
import { Link } from 'umi';
import { useTranslation } from 'react-i18next';
import { INTRO_THEME } from '@/components/creator/creatorTheme';
import { WorkflowGraph } from '@/components/creator/WorkflowGraph';
import { MemoryGraphThree } from '@/components/creator/MemoryGraphThree';
import { IntroFooter } from '@/components/creator/IntroFooter';
import { MEMORY_GRAPH_DATA } from '@/components/creator/memoryGraphData';

const T = INTRO_THEME;
const { Title, Paragraph, Text } = Typography;

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.032, delayChildren: 0.15 } },
};
const charItem = { hidden: { y: '1em', opacity: 0 }, visible: { y: 0, opacity: 1 } };

export default function IntroPage() {
  const { t } = useTranslation();
  const [hoverCard, setHoverCard] = useState<number | null>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0.5);
  const mouseY = useMotionValue(0.5);

  const workflowSteps = useMemo(
    () => [
      {
        id: 'ideate',
        title: t('intro.stepIdeateTitle'),
        subtitle: t('intro.stepIdeateSubtitle'),
        description: t('intro.stepIdeateDesc'),
        icon: <BulbOutlined />,
        color: '#3b82f6',
        features: t('intro.stepIdeateFeatures', { returnObjects: true }) as string[],
      },
      {
        id: 'create',
        title: t('intro.stepCreateTitle'),
        subtitle: t('intro.stepCreateSubtitle'),
        description: t('intro.stepCreateDesc'),
        icon: <EditOutlined />,
        color: '#10b981',
        features: t('intro.stepCreateFeatures', { returnObjects: true }) as string[],
      },
      {
        id: 'optimize',
        title: t('intro.stepOptimizeTitle'),
        subtitle: t('intro.stepOptimizeSubtitle'),
        description: t('intro.stepOptimizeDesc'),
        icon: <RiseOutlined />,
        color: '#f59e0b',
        features: t('intro.stepOptimizeFeatures', { returnObjects: true }) as string[],
      },
    ],
    [t]
  );
  const titleLine1 = t('intro.titleLine1');
  const titleLine2 = t('intro.titleLine2');

  useEffect(() => {
    const el = heroRef.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      mouseX.set((e.clientX - rect.left) / rect.width);
      mouseY.set((e.clientY - rect.top) / rect.height);
    };
    window.addEventListener('mousemove', onMove, { passive: true });
    return () => window.removeEventListener('mousemove', onMove);
  }, [mouseX, mouseY]);

  const orb1X = useTransform(mouseX, [0, 1], ['2%', '8%']);
  const orb1Y = useTransform(mouseY, [0, 1], ['10%', '20%']);
  const orb2X = useTransform(mouseX, [0, 1], ['88%', '94%']);
  const orb2Y = useTransform(mouseY, [0, 1], ['60%', '75%']);

  const [graphSize, setGraphSize] = useState({ w: 1080, h: 560 });
  useEffect(() => {
    const update = () => setGraphSize({ w: Math.min(1200, window.innerWidth - 80), h: 560 });
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return (
    <div
      className="intro-page"
      style={{
        fontFamily: T.fontFamily,
        color: T.text,
        background: T.bgPage,
        minHeight: '100%',
      }}
    >
      <style>{`
        .intro-page .intro-gradient-bg {
          background: linear-gradient(135deg,
            rgba(255,255,255,1) 0%,
            rgba(255,75,47,0.08) 20%,
            rgba(255,167,167,0.06) 40%,
            rgba(255,178,158,0.08) 60%,
            rgba(255,255,255,1) 100%);
          background-size: 400% 400%;
          animation: intro-gradient-shift 20s ease infinite;
        }
        @keyframes intro-gradient-shift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .intro-page .intro-noise::before {
          content: '';
          position: absolute;
          inset: 0;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
          opacity: 0.035;
          pointer-events: none;
        }
        .intro-page .intro-float { animation: intro-float 5s ease-in-out infinite; }
        .intro-page .intro-float-delay { animation: intro-float 5s ease-in-out infinite; animation-delay: 2.5s; }
        @keyframes intro-float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-12px); }
        }
        .intro-page .intro-shimmer {
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
          background-size: 200% 100%;
          animation: intro-shimmer 2.5s ease-in-out infinite;
        }
        @keyframes intro-shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .intro-page .intro-char { display: inline-block; overflow: hidden; }
        .intro-page .intro-char-inner { display: inline-block; }
        .intro-page .intro-ping {
          animation: intro-ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;
        }
        @keyframes intro-ping {
          75%, 100% { transform: scale(1.8); opacity: 0; }
        }
        .intro-page .intro-text-gradient {
          background: linear-gradient(135deg, #ff4b2f 0%, #ffa7a7 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .intro-page .intro-steps-container {
          flex-direction: column;
        }
        .intro-page .intro-step-card {
          min-width: 0;
        }
        @media (min-width: 900px) {
          .intro-page .intro-steps-container {
            flex-direction: row;
            min-height: 520px;
          }
          .intro-page .intro-step-card {
            min-height: 480px !important;
          }
        }
        .intro-page .intro-footer-grid {
          display: grid;
          grid-template-columns: 2fr repeat(4, 1fr);
          gap: 32px 40px;
          align-items: start;
        }
        .intro-page .intro-footer-brand {
          grid-column: 1;
        }
        @media (max-width: 799px) {
          .intro-page .intro-footer-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 24px 28px;
          }
          .intro-page .intro-footer-brand {
            grid-column: 1 / -1;
          }
        }
        .intro-page .intro-footer-link:hover {
          color: #fff !important;
        }
        .intro-page .intro-footer-social:hover {
          background: #ff4b2f !important;
          color: #fff !important;
        }
        .intro-page .intro-footer-email input.ant-input::placeholder {
          color: rgba(255,255,255,0.4);
        }
        .intro-page .intro-footer-email input.ant-input {
          color: #fff;
          background: transparent;
        }
        .intro-page .intro-footer-email.ant-input-affix-wrapper,
        .intro-page .intro-footer-email.ant-input {
          background: rgba(255,255,255,0.06) !important;
          border-color: rgba(255,255,255,0.15);
        }
        .intro-page .intro-footer-email.ant-input:focus,
        .intro-page .intro-footer-email.ant-input-focused,
        .intro-page .intro-footer-email.ant-input-affix-wrapper-focused {
          border-color: #ff4b2f !important;
          box-shadow: 0 0 0 2px rgba(255,75,47,0.2);
        }
        .intro-page .intro-footer-newsletter {
          flex-wrap: nowrap;
        }
        .intro-page .intro-footer-bottom {
          flex-wrap: nowrap;
        }
        @media (max-width: 639px) {
          .intro-page .intro-footer-newsletter {
            flex-wrap: wrap;
          }
          .intro-page .intro-footer-bottom {
            flex-wrap: wrap;
          }
        }
      `}</style>

      {/* Hero */}
      <section
        ref={heroRef}
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '80px 40px 100px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div className="intro-gradient-bg" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />
        <div className="intro-noise" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />
        <motion.div
          className="intro-float"
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
        />
        <motion.div
          className="intro-float-delay"
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
        />

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          style={{ maxWidth: 960, textAlign: 'center', position: 'relative', zIndex: 1 }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1, duration: 0.5, type: 'spring', stiffness: 200 }}
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
              <span className="intro-ping" style={{ position: 'absolute', width: 8, height: 8, borderRadius: '50%', background: T.accent, opacity: 0.75 }} />
              <span style={{ position: 'relative', width: 6, height: 6, borderRadius: '50%', background: T.accent }} />
            </span>
            <ThunderboltOutlined style={{ fontSize: 15 }} />
            {t('intro.badge')}
          </motion.div>

          <motion.h1
            variants={container}
            initial="hidden"
            animate="visible"
            style={{
              color: T.textBright,
              fontWeight: T.fontWeightBold,
              fontSize: 'clamp(32px, 5.5vw, 56px)',
              lineHeight: 1.25,
              marginBottom: 16,
              margin: '0 auto 16px',
            }}
          >
            {titleLine1.split('').map((c, i) => (
              <motion.span key={`1-${i}`} className="intro-char" variants={charItem}>
                <span className="intro-char-inner">{c === ' ' ? '\u00A0' : c}</span>
              </motion.span>
            ))}
            <br />
            {titleLine2.split('').map((c, i) => (
              <motion.span key={`2-${i}`} className="intro-char" variants={charItem} style={{ color: T.accent }}>
                <span className="intro-char-inner">{c === ' ' ? '\u00A0' : c}</span>
              </motion.span>
            ))}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, filter: 'blur(10px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            transition={{ duration: 0.9, delay: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ color: T.textMuted, fontSize: 'clamp(16px, 2.2vw, 19px)', lineHeight: 1.7, marginBottom: 40, maxWidth: 672, margin: '0 auto 40px', letterSpacing: '0.02em' }}
          >
            {t('intro.subtitle')}
          </motion.p>

          <motion.div
            initial="hidden"
            animate="visible"
            variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.08, delayChildren: 0.6 } } }}
            style={{ display: 'flex', flexWrap: 'wrap', gap: 18, justifyContent: 'center' }}
          >
            <motion.div variants={{ hidden: { scale: 0, opacity: 0 }, visible: { scale: 1, opacity: 1 } }} transition={{ type: 'spring', stiffness: 260, damping: 18 }} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
              <Link to="/creator">
                <Button type="primary" size="large" icon={<ThunderboltOutlined />} style={{ background: T.primaryBg, borderColor: T.primaryBg, height: 52, paddingLeft: 28, paddingRight: 28, fontSize: 16, fontWeight: T.fontWeightSemibold, borderRadius: T.radiusMd, boxShadow: '0 4px 14px rgba(255,75,47,0.35)' }}>
                  {t('intro.ctaCreator')}
                </Button>
              </Link>
            </motion.div>
            <motion.div variants={{ hidden: { scale: 0, opacity: 0 }, visible: { scale: 1, opacity: 1 } }} transition={{ type: 'spring', stiffness: 260, damping: 18 }} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
              <Link to="/ai">
                <Button size="large" icon={<MessageOutlined />} style={{ height: 52, paddingLeft: 28, paddingRight: 28, fontSize: 16, color: T.text, border: `2px solid ${T.stroke}`, background: 'transparent', fontWeight: T.fontWeightMedium, borderRadius: T.radiusMd }}>
                  {t('intro.ctaAI')}
                </Button>
              </Link>
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1 }}
            style={{ display: 'flex', flexWrap: 'wrap', gap: 48, justifyContent: 'center', marginTop: 48 }}
          >
            {[
              { value: t('intro.statsMultiAgent'), label: t('intro.statsMultiAgentLabel') },
              { value: t('intro.statsCloudMemory'), label: t('intro.statsCloudMemoryLabel') },
              { value: t('intro.statsPipeline'), label: t('intro.statsPipelineLabel') },
            ].map((s, i) => (
              <motion.div key={i} style={{ textAlign: 'center' }} whileHover={{ y: -2 }} transition={{ duration: 0.2 }}>
                <div style={{ fontSize: 18, fontWeight: T.fontWeightBold, color: T.textBright }}>{s.value}</div>
                <div style={{ fontSize: 13, color: T.textDim }}>{s.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </section>

      {/* 实时查看工作流 */}
      <section style={{ padding: '120px 40px 140px', background: 'linear-gradient(180deg, #fafafa 0%, #ffffff 100%)', minHeight: 'min(100vh, 900px)' }}>
        <div style={{ maxWidth: 1280, margin: '0 auto' }}>
          <motion.div
            initial={{ opacity: 0, y: 32 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ textAlign: 'center', marginBottom: 56 }}
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
              <span style={{ position: 'relative', display: 'inline-flex', width: 12, height: 12, alignItems: 'center', justifyContent: 'center' }}>
                <span className="intro-ping" style={{ position: 'absolute', width: 6, height: 6, borderRadius: '50%', background: T.accent, opacity: 0.75 }} />
                <span style={{ position: 'relative', width: 5, height: 5, borderRadius: '50%', background: T.accent }} />
              </span>
              <span style={{ fontSize: 14, fontWeight: T.fontWeightSemibold, color: T.accent }}>{t('intro.workflowLive')}</span>
            </div>
            <Title level={2} style={{ color: T.textBright, fontWeight: T.fontWeightBold, marginBottom: 12, fontSize: 32 }}>
              {t('intro.workflowTitle')} <span className="intro-text-gradient">{t('intro.workflowTitleHighlight')}</span>
            </Title>
            <Paragraph style={{ color: T.textMuted, fontSize: 18, maxWidth: 672, margin: '0 auto' }}>
              {t('intro.workflowDesc')}
            </Paragraph>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 40, scale: 0.98 }}
            whileInView={{ opacity: 1, y: 0, scale: 1 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ marginBottom: 48 }}
          >
            <div style={{ maxWidth: 1240, margin: '0 auto' }}>
              <WorkflowGraph
                variant="creation"
                demo
                showAgentCenter
                scale={1.1}
                height={480}
                width={950}
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 24, marginBottom: 48 }}
          >
            {[
              { icon: <ApiOutlined />, value: t('intro.stat24_7'), label: t('intro.stat24_7Label'), color: T.accent },
              { icon: <ThunderboltOutlined />, value: t('intro.statResponse'), label: t('intro.statResponseLabel'), color: '#10b981' },
              { icon: <CheckCircleOutlined />, value: t('intro.statSuccess'), label: t('intro.statSuccessLabel'), color: '#3b82f6' },
              { icon: <RocketOutlined />, value: t('intro.statEfficiency'), label: t('intro.statEfficiencyLabel'), color: '#f59e0b' },
            ].map((s, i) => (
              <div
                key={i}
                style={{
                  padding: 28,
                  borderRadius: T.radiusLg,
                  background: 'rgba(248,248,248,0.9)',
                  border: `1px solid ${T.border}`,
                  textAlign: 'center',
                  transition: 'box-shadow 0.25s, transform 0.25s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = '0 12px 24px rgba(0,0,0,0.06)';
                  e.currentTarget.style.transform = 'translateY(-4px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = 'none';
                  e.currentTarget.style.transform = 'none';
                }}
              >
                <div style={{ width: 52, height: 52, borderRadius: 12, background: `${s.color}18`, color: s.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, margin: '0 auto 14px' }}>
                  {s.icon}
                </div>
                <div style={{ fontSize: 24, fontWeight: T.fontWeightBold, color: T.textBright, marginBottom: 4 }}>{s.value}</div>
                <div style={{ fontSize: 14, color: T.textDim }}>{s.label}</div>
              </div>
            ))}
          </motion.div>

          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} style={{ textAlign: 'center' }}>
            <Link to="/creator">
              <Button type="primary" size="large" icon={<ArrowRightOutlined />} style={{ background: T.primaryBg, borderColor: T.primaryBg, height: 52, paddingLeft: 32, paddingRight: 32, fontSize: 16, fontWeight: T.fontWeightSemibold, borderRadius: T.radiusMd }}>
                {t('intro.ctaWorkflow')}
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* 流程：构思 → 创作 → 优化 — 参考 WorkflowCards 手风琴 + 动效 */}
      <section style={{ padding: '100px 40px 120px', maxWidth: 1280, margin: '0 auto', minHeight: 'min(100vh, 820px)', background: T.bgPage }}>
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          style={{ textAlign: 'center', marginBottom: 56 }}
        >
          <Title level={2} style={{ color: T.textBright, fontWeight: T.fontWeightBold, marginBottom: 8, fontSize: 32 }}>
            {t('intro.stepsTitle')} <span className="intro-text-gradient">{t('intro.stepsTitleHighlight')}</span>
          </Title>
          <Text style={{ color: T.textMuted, fontSize: 18 }}>{t('intro.stepsSubtitle')}</Text>
        </motion.div>

        <motion.div
          layout
          style={{
            display: 'flex',
            gap: 20,
            minHeight: 420,
          }}
          className="intro-steps-container"
        >
          {workflowSteps.map((step, index) => {
            const isActive = hoverCard === index;
            const isCollapsed = hoverCard !== null && hoverCard !== index;
            return (
              <motion.div
                key={step.id}
                layout
                initial={{ opacity: 0, y: 36 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ delay: index * 0.12, duration: 0.55, ease: [0.25, 0.46, 0.45, 0.94] }}
                onMouseEnter={() => setHoverCard(index)}
                onMouseLeave={() => setHoverCard(null)}
                style={{
                  flex: isActive ? 2 : isCollapsed ? 0.6 : 1,
                  minHeight: isActive ? 360 : isCollapsed ? 140 : 260,
                  display: 'flex',
                  flexDirection: 'column',
                  position: 'relative',
                  borderRadius: T.radiusLg,
                  overflow: 'hidden',
                  cursor: 'pointer',
                  transition: 'flex 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94), min-height 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                }}
                className="intro-step-card"
              >
                {/* 背景：激活时渐变 */}
                <motion.div
                  layout
                  transition={{ duration: 0.4 }}
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: `linear-gradient(135deg, ${step.color}18 0%, ${step.color}08 100%)`,
                    opacity: isActive ? 1 : 0,
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'rgba(248,248,248,0.95)',
                    opacity: isActive ? 0 : 1,
                    transition: 'opacity 0.4s',
                  }}
                />
                {/* 激活边框 */}
                <motion.div
                  layout
                  transition={{ duration: 0.3 }}
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: T.radiusLg,
                    border: `2px solid ${step.color}`,
                    opacity: isActive ? 1 : 0,
                    pointerEvents: 'none',
                  }}
                />

                <div style={{ position: 'relative', flex: 1, padding: 24, display: 'flex', flexDirection: 'column' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
                    <motion.div
                      animate={isActive ? { scale: 1.1 } : { scale: 1 }}
                      transition={{ type: 'spring', stiffness: 300, damping: 22 }}
                      style={{
                        width: 56,
                        height: 56,
                        borderRadius: T.radiusMd,
                        background: isActive ? step.color : `${step.color}18`,
                        color: isActive ? '#fff' : step.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 26,
                        flexShrink: 0,
                      }}
                    >
                      {step.icon}
                    </motion.div>
                    <motion.span
                      animate={{ opacity: isActive ? 0.25 : 0.12 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        fontSize: 56,
                        fontWeight: T.fontWeightBold,
                        color: step.color,
                        lineHeight: 1,
                        flexShrink: 0,
                      }}
                    >
                      0{index + 1}
                    </motion.span>
                  </div>

                  <div style={{ marginBottom: 8 }}>
                    <Text style={{ color: T.textDim, fontSize: 13, fontWeight: T.fontWeightMedium, display: 'block', marginBottom: 4 }}>{step.subtitle}</Text>
                    <Title level={4} style={{ color: T.textBright, margin: 0, fontWeight: T.fontWeightSemibold, fontSize: 24 }}>{step.title}</Title>
                  </div>

                  <motion.div
                    animate={{
                      opacity: isActive ? 1 : 0,
                      maxHeight: isActive ? 320 : 0,
                    }}
                    transition={{ duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
                    style={{ overflow: 'hidden', flex: 1 }}
                  >
                    <Paragraph style={{ color: T.textMuted, fontSize: 16, lineHeight: 1.6, marginBottom: 14 }}>
                      {step.description}
                    </Paragraph>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                      {step.features.map((f, i) => (
                        <motion.span
                          key={i}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={isActive ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
                          transition={{ delay: 0.05 * i, duration: 0.25 }}
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
                        </motion.span>
                      ))}
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      </section>

      {/* 探索你的记忆系统 — 深色区块 */}
      <section
        style={{
          padding: '120px 40px 140px',
          background: 'linear-gradient(180deg, #0a0a0a 0%, #0f0f1a 40%, #1a1a2e 100%)',
          position: 'relative',
          overflow: 'hidden',
          minHeight: 'min(100vh, 920px)',
        }}
      >
        <div style={{ position: 'absolute', top: '20%', left: '15%', width: 280, height: 280, borderRadius: '50%', background: 'rgba(255,75,47,0.06)', filter: 'blur(60px)', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', bottom: '20%', right: '15%', width: 240, height: 240, borderRadius: '50%', background: 'rgba(59,130,246,0.06)', filter: 'blur(50px)', pointerEvents: 'none' }} />
        <div
          style={{
            position: 'absolute',
            inset: 0,
            opacity: 0.08,
            pointerEvents: 'none',
            backgroundImage: 'linear-gradient(to right, #ff4b2f 1px, transparent 1px), linear-gradient(to bottom, #ff4b2f 1px, transparent 1px)',
            backgroundSize: '48px 48px',
          }}
        />

        <div style={{ maxWidth: 1280, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <motion.div
            initial={{ opacity: 0, y: 32 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ textAlign: 'center', marginBottom: 40 }}
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
              <DatabaseOutlined style={{ color: T.accent, fontSize: 14 }} />
              <span style={{ fontSize: 13, fontWeight: T.fontWeightBold, color: T.accent, letterSpacing: '0.04em' }}>{t('intro.memoryBadge')}</span>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e' }} />
            </div>
            <Title level={2} style={{ color: '#fafafa', fontWeight: T.fontWeightBold, marginBottom: 12, fontSize: 32 }}>
              {t('intro.memoryTitle')} <span className="intro-text-gradient">{t('intro.memoryTitleHighlight')}</span>
            </Title>
            <Paragraph style={{ color: 'rgba(255,255,255,0.55)', fontSize: 18, maxWidth: 672, margin: '0 auto' }}>
              {t('intro.memoryDesc')}
            </Paragraph>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ borderRadius: 16, overflow: 'hidden', marginBottom: 36, boxShadow: '0 0 0 1px rgba(255,255,255,0.08), 0 24px 48px rgba(0,0,0,0.4)' }}
          >
            <MemoryGraphThree data={MEMORY_GRAPH_DATA} width={graphSize.w} height={graphSize.h} onNodeClick={() => {}} variant="intro" />
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 40 }}
          >
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: T.accent }} />
            <span style={{ fontSize: 14, color: 'rgba(255,255,255,0.45)' }}>{t('intro.memoryHint')}</span>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} style={{ textAlign: 'center' }}>
            <Link to="/creator">
              <Button type="primary" size="large" icon={<ArrowRightOutlined />} style={{ background: T.primaryBg, borderColor: T.primaryBg, height: 52, paddingLeft: 32, paddingRight: 32, fontSize: 16, fontWeight: T.fontWeightSemibold, borderRadius: T.radiusMd }}>
                {t('intro.ctaMemory')}
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ padding: '100px 40px 120px', textAlign: 'center', minHeight: 'min(80vh, 520px)', background: 'linear-gradient(180deg, #ffffff 0%, #fafafa 100%)' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          style={{
            maxWidth: 720,
            margin: '0 auto',
            padding: '64px 48px',
            borderRadius: T.radiusXl,
            background: T.accentDim,
            border: `1px solid ${T.accentBorder}`,
            boxShadow: T.shadowCta,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div className="intro-shimmer" style={{ position: 'absolute', inset: 0, pointerEvents: 'none', borderRadius: 'inherit' }} />
          <div style={{ position: 'relative', zIndex: 1 }}>
            <Title level={3} style={{ color: T.textBright, marginBottom: 12, fontWeight: T.fontWeightBold, fontSize: 28 }}>
              {t('intro.finalTitle')}
            </Title>
            <Paragraph style={{ color: T.textMuted, marginBottom: 28, fontSize: 18, maxWidth: 560, margin: '0 auto 28px' }}>
              {t('intro.finalSubtitle')}
            </Paragraph>
            <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
              <Link to="/creator">
                <Button type="primary" size="large" icon={<ArrowRightOutlined />} style={{ background: T.primaryBg, borderColor: T.primaryBg, height: 52, paddingLeft: 32, paddingRight: 32, fontSize: 16, fontWeight: T.fontWeightSemibold, borderRadius: T.radiusMd }}>
                  {t('intro.finalCta')}
                </Button>
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </section>

      <IntroFooter />
    </div>
  );
}

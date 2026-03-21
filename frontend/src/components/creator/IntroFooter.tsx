/**
 * 主页页脚 — 与参考一致：深色、品牌 + 四栏链接 + 订阅 + 底栏
 */
import React from 'react';
import { Button, Input } from 'antd';
import {
  ThunderboltOutlined,
  MailOutlined,
  PhoneOutlined,
  EnvironmentOutlined,
  TwitterOutlined,
  GithubOutlined,
  LinkedinOutlined,
  YoutubeOutlined,
} from '@ant-design/icons';
import { Link } from 'umi';
import { INTRO_THEME } from './creatorTheme';

const T = INTRO_THEME;

const FOOTER_LINKS = {
  product: {
    title: '产品',
    links: [
      { name: '主页', href: '/' },
      { name: '创作助手', href: '/creator' },
      { name: 'AI 对话', href: '/ai' },
      { name: '定价方案', href: '#pricing' },
      { name: '更新日志', href: '#changelog' },
    ],
  },
  resources: {
    title: '资源',
    links: [
      { name: '帮助中心', href: '#help' },
      { name: '使用教程', href: '#tutorials' },
      { name: 'API 文档', href: '#api' },
      { name: '博客', href: '#blog' },
      { name: '社区', href: '#community' },
    ],
  },
  company: {
    title: '公司',
    links: [
      { name: '关于我们', href: '#about' },
      { name: '加入我们', href: '#careers' },
      { name: '联系我们', href: '#contact' },
      { name: '合作伙伴', href: '#partners' },
      { name: '媒体资源', href: '#press' },
    ],
  },
  legal: {
    title: '法律',
    links: [
      { name: '隐私政策', href: '#privacy' },
      { name: '服务条款', href: '#terms' },
      { name: 'Cookie 政策', href: '#cookies' },
      { name: '安全合规', href: '#security' },
    ],
  },
} as const;

const SOCIAL_LINKS = [
  { icon: <TwitterOutlined />, href: '#', label: 'Twitter' },
  { icon: <GithubOutlined />, href: '#', label: 'GitHub' },
  { icon: <LinkedinOutlined />, href: '#', label: 'LinkedIn' },
  { icon: <YoutubeOutlined />, href: '#', label: 'YouTube' },
] as const;

type Section = { title: string; links: readonly { name: string; href: string }[] };

export const IntroFooter: React.FC = () => (
  <footer
    className="intro-footer"
    style={{
      background: '#0f0f0f',
      color: '#fff',
      fontFamily: T.fontFamily,
    }}
  >
    <div style={{ maxWidth: 1280, margin: '0 auto', padding: '64px 40px 0' }}>
      <div className="intro-footer-grid">
        <div className="intro-footer-brand">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 24 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 12,
                background: 'linear-gradient(135deg, #ff4b2f 0%, #ffa7a7 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontSize: 18,
              }}
            >
              <ThunderboltOutlined />
            </div>
            <span style={{ fontSize: 20, fontWeight: T.fontWeightBold }}>创想AI</span>
          </div>
          <p style={{ color: 'rgba(255,255,255,0.5)', marginBottom: 24, maxWidth: 320, lineHeight: 1.6, fontSize: 14 }}>
            智能创作助手，让 AI 为你的创意赋能。从构思到发布，一站式内容创作平台。
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: 'rgba(255,255,255,0.5)' }}>
            <a href="mailto:decalogue80@gmail.com" style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'inherit', textDecoration: 'none', transition: 'color 0.2s' }} className="intro-footer-link">
              <MailOutlined style={{ fontSize: 14 }} />
              <span>decalogue80@gmail.com</span>
            </a>
            <a href="tel:13661445290" style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'inherit', textDecoration: 'none', transition: 'color 0.2s' }} className="intro-footer-link">
              <PhoneOutlined style={{ fontSize: 14 }} />
              <span>13661445290</span>
            </a>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <EnvironmentOutlined style={{ fontSize: 14 }} />
              <span>上海市</span>
            </div>
          </div>
        </div>

        {(Object.values(FOOTER_LINKS) as Section[]).map((section) => (
          <div key={section.title}>
            <h4 style={{ fontSize: 14, fontWeight: T.fontWeightSemibold, color: '#fff', marginBottom: 16 }}>
              {section.title}
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
              {section.links.map((link) => (
                <li key={link.name}>
                  {link.href.startsWith('/') ? (
                    <Link
                      to={link.href}
                      style={{
                        color: 'rgba(255,255,255,0.5)',
                        fontSize: 13,
                        textDecoration: 'none',
                        transition: 'color 0.2s',
                      }}
                      className="intro-footer-link"
                    >
                      {link.name}
                    </Link>
                  ) : (
                    <a
                      href={link.href}
                      style={{
                        color: 'rgba(255,255,255,0.5)',
                        fontSize: 13,
                        textDecoration: 'none',
                        transition: 'color 0.2s',
                      }}
                      className="intro-footer-link"
                    >
                      {link.name}
                    </a>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 48,
          paddingTop: 24,
          borderTop: '1px solid rgba(255,255,255,0.1)',
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 24,
        }}
        className="intro-footer-newsletter"
      >
        <div className="intro-footer-newsletter-left" style={{ flex: '1 1 200px', minWidth: 0, textAlign: 'left' }}>
          <h4 style={{ fontSize: 15, fontWeight: T.fontWeightSemibold, color: '#fff', marginBottom: 6 }}>
            订阅我们的新闻
          </h4>
          <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, margin: 0 }}>获取最新产品更新和创作技巧</p>
        </div>
        <div className="intro-footer-newsletter-right" style={{ display: 'flex', gap: 12, flexShrink: 0, alignItems: 'center' }}>
          <Input
            placeholder="输入你的邮箱"
            style={{
              width: 240,
              minWidth: 180,
              height: 44,
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 10,
              color: '#fff',
            }}
            className="intro-footer-email"
          />
          <Button
            type="primary"
            style={{
              height: 44,
              paddingLeft: 24,
              paddingRight: 24,
              background: T.primaryBg,
              borderColor: T.primaryBg,
              borderRadius: 10,
              fontWeight: T.fontWeightMedium,
            }}
          >
            订阅
          </Button>
        </div>
      </div>
    </div>

    <div
      style={{
        borderTop: '1px solid rgba(255,255,255,0.1)',
        marginTop: 24,
      }}
    >
      <div
        style={{
          maxWidth: 1280,
          margin: '0 auto',
          padding: '20px 40px',
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 16,
        }}
        className="intro-footer-bottom"
      >
        <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, margin: 0, flex: '1 1 auto', minWidth: 0 }}>
          © 2026 创想AI. 保留所有权利.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
          {SOCIAL_LINKS.map((s) => (
            <a
              key={s.label}
              href={s.href}
              aria-label={s.label}
              style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background: 'rgba(255,255,255,0.06)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'rgba(255,255,255,0.5)',
                fontSize: 18,
                transition: 'background 0.2s, color 0.2s',
              }}
              className="intro-footer-social"
            >
              {s.icon}
            </a>
          ))}
        </div>
      </div>
    </div>
  </footer>
);

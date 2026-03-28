/**
 * 站点页脚：品牌、复习产品链接、订阅区（演示）
 */
import React, { useMemo } from 'react';
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
import { useTranslation } from 'react-i18next';
import { INTRO_THEME } from './creatorTheme';

const T = INTRO_THEME;

const SOCIAL_LINKS = [
  { icon: <TwitterOutlined />, href: '#', label: 'Twitter' },
  { icon: <GithubOutlined />, href: '#', label: 'GitHub' },
  { icon: <LinkedinOutlined />, href: '#', label: 'LinkedIn' },
  { icon: <YoutubeOutlined />, href: '#', label: 'YouTube' },
] as const;

type Section = { title: string; links: readonly { name: string; href: string }[] };

export const IntroFooter: React.FC = () => {
  const { t } = useTranslation();

  const FOOTER_LINKS = useMemo(
    () =>
      ({
        product: {
          title: t('footer.sectionProduct'),
          links: [
            { name: t('footer.linkHome'), href: '/' },
            { name: t('footer.linkReviewHub'), href: '/review' },
            { name: t('footer.linkReviewToday'), href: '/review/today' },
            { name: t('footer.linkStars'), href: '/stars' },
            { name: t('footer.linkAiHub'), href: '/ai' },
            { name: t('footer.linkChat'), href: '/chat' },
          ],
        },
        resources: {
          title: t('footer.sectionResources'),
          links: [
            { name: t('footer.linkHelp'), href: '#help' },
            { name: t('footer.linkTutorials'), href: '#tutorials' },
            { name: t('footer.linkApi'), href: '#api' },
            { name: t('footer.linkBlog'), href: '#blog' },
            { name: t('footer.linkCommunity'), href: '#community' },
          ],
        },
        company: {
          title: t('footer.sectionCompany'),
          links: [
            { name: t('footer.linkAbout'), href: '#about' },
            { name: t('footer.linkCareers'), href: '#careers' },
            { name: t('footer.linkContact'), href: '#contact' },
            { name: t('footer.linkPartners'), href: '#partners' },
            { name: t('footer.linkPress'), href: '#press' },
          ],
        },
        legal: {
          title: t('footer.sectionLegal'),
          links: [
            { name: t('footer.linkPrivacy'), href: '#privacy' },
            { name: t('footer.linkTerms'), href: '#terms' },
            { name: t('footer.linkCookies'), href: '#cookies' },
            { name: t('footer.linkSecurity'), href: '#security' },
          ],
        },
      }) as const,
    [t]
  );

  return (
    <footer
      className="site-footer"
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
              <span style={{ fontSize: 20, fontWeight: T.fontWeightBold }}>{t('footer.brand')}</span>
            </div>
            <p style={{ color: 'rgba(255,255,255,0.5)', marginBottom: 24, maxWidth: 320, lineHeight: 1.6, fontSize: 14 }}>
              {t('footer.tagline')}
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
                <span>{t('footer.location')}</span>
              </div>
            </div>
          </div>

          {(Object.values(FOOTER_LINKS) as Section[]).map((section) => (
            <div key={section.title}>
              <h4 style={{ fontSize: 14, fontWeight: T.fontWeightSemibold, color: '#fff', marginBottom: 16 }}>{section.title}</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {section.links.map((link) => (
                  <li key={link.name + link.href}>
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
            <h4 style={{ fontSize: 15, fontWeight: T.fontWeightSemibold, color: '#fff', marginBottom: 6 }}>{t('footer.newsletterTitle')}</h4>
            <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, margin: 0 }}>{t('footer.newsletterDesc')}</p>
          </div>
          <div className="intro-footer-newsletter-right" style={{ display: 'flex', gap: 12, flexShrink: 0, alignItems: 'center' }}>
            <Input
              placeholder={t('footer.emailPlaceholder')}
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
              {t('footer.subscribe')}
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
            {t('footer.copyright')}
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
};

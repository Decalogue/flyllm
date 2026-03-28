import React, { useMemo } from 'react';
import { Link, Outlet, useLocation } from 'umi';
import { Space } from 'antd';
import { ThunderboltOutlined, MessageOutlined, HomeOutlined, ReadOutlined, StarOutlined, CommentOutlined } from '@ant-design/icons';
import { CREATOR_THEME, INTRO_THEME } from '@/components/creator/creatorTheme';
import { LocaleSwitcher } from '@/components/creator/LocaleSwitcher';
import { useTranslation } from 'react-i18next';

export default function Layout() {
  const { pathname } = useLocation();
  const { t } = useTranslation();

  const isLightHeader = pathname === '/' || pathname.startsWith('/review');
  const LT = isLightHeader ? INTRO_THEME : CREATOR_THEME;

  const navItems = useMemo(
    () => [
      { path: '/', label: t('layout.navHome'), icon: <HomeOutlined /> },
      { path: '/review', label: t('layout.navReview'), icon: <ReadOutlined /> },
      { path: '/stars', label: t('layout.navStars'), icon: <StarOutlined /> },
      { path: '/ai', label: t('layout.navAi'), icon: <MessageOutlined /> },
      { path: '/chat', label: t('layout.navChat'), icon: <CommentOutlined /> },
    ],
    [t]
  );

  const headerBg = isLightHeader ? 'rgba(255,255,255,0.9)' : CREATOR_THEME.bgHeader;
  const headerBorder = isLightHeader ? INTRO_THEME.border : CREATOR_THEME.borderHeader;
  const headerShadow = isLightHeader ? '0 0 0 1px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.04)' : CREATOR_THEME.shadowPanel;
  const logoColor = isLightHeader ? INTRO_THEME.textBright : CREATOR_THEME.textBright;

  return (
    <div
      className="app-layout"
      data-theme={isLightHeader ? 'intro' : 'creator'}
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: isLightHeader ? INTRO_THEME.bgPage : CREATOR_THEME.bgPage,
        fontFamily: CREATOR_THEME.fontFamily,
        color: LT.text,
        ['--nav-inactive' as string]: isLightHeader ? INTRO_THEME.textMuted : CREATOR_THEME.textMuted,
        ['--nav-active-color' as string]: isLightHeader ? INTRO_THEME.accent : CREATOR_THEME.segSelectedText,
        ['--nav-active-bg' as string]: isLightHeader ? INTRO_THEME.accentDim : CREATOR_THEME.segSelectedBg,
        ['--nav-hover-bg' as string]: isLightHeader ? 'rgba(0,0,0,0.04)' : CREATOR_THEME.segBg,
        ['--nav-hover-color' as string]: isLightHeader ? INTRO_THEME.text : CREATOR_THEME.text,
      }}
    >
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          background: headerBg,
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${headerBorder}`,
          boxShadow: headerShadow,
        }}
      >
        <div
          style={{
            maxWidth: 1200,
            margin: '0 auto',
            padding: '12px 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Link
            to="/"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              color: logoColor,
              fontWeight: CREATOR_THEME.fontWeightBold,
              fontSize: 18,
              textDecoration: 'none',
            }}
          >
            <span
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background: `linear-gradient(135deg, ${CREATOR_THEME.brand}, ${CREATOR_THEME.brandLight})`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontSize: 16,
              }}
            >
              <ThunderboltOutlined />
            </span>
            {t('layout.appName')}
          </Link>

          <Space size="middle" style={{ flex: 1, justifyContent: 'flex-end', alignItems: 'center', flexWrap: 'wrap' }}>
            {navItems.map(({ path, label, icon }) => {
              const isActive = pathname === path || (path !== '/' && pathname.startsWith(path));
              return (
                <Link
                  key={path}
                  to={path}
                  className={`app-layout-nav-link${isActive ? ' app-layout-nav-link--active' : ''}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '8px 14px',
                    borderRadius: CREATOR_THEME.radiusSm,
                    fontSize: 14,
                    textDecoration: 'none',
                    transition: 'color 0.2s, background 0.2s',
                  }}
                >
                  {icon}
                  {label}
                </Link>
              );
            })}
            <LocaleSwitcher size="small" />
          </Space>
        </div>
      </header>

      <main
        style={{
          flex: 1,
          overflow: pathname === '/' || pathname.startsWith('/review') ? 'auto' : 'hidden',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}
      >
        <Outlet />
      </main>
    </div>
  );
}

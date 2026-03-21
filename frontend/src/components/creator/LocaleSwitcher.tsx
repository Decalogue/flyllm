/**
 * 语言切换：中 / EN，置于布局顶栏右侧，持久化到 localStorage
 */
import React from 'react';
import { Select } from 'antd';
import { useTranslation } from 'react-i18next';
import { setStoredLocale, type LocaleType } from '@/utils/i18n';

const LOCALE_OPTIONS: { value: LocaleType; label: string }[] = [
  { value: 'zh-CN', label: '中' },
  { value: 'en-US', label: 'EN' },
];

export const LocaleSwitcher: React.FC<{
  size?: 'small' | 'middle' | 'large';
  style?: React.CSSProperties;
  className?: string;
}> = ({ size = 'middle', style, className }) => {
  const { i18n } = useTranslation();
  const value = (i18n.language === 'en-US' ? 'en-US' : 'zh-CN') as LocaleType;

  return (
    <Select
      value={value}
      options={LOCALE_OPTIONS}
      onChange={(v) => setStoredLocale(v as LocaleType)}
      size={size}
      style={{ minWidth: 56, ...style }}
      className={className}
    />
  );
};

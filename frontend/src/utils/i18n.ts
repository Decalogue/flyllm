/**
 * i18n 初始化：react-i18next + 中英切换
 * 语言存 localStorage key: creator_locale (zh-CN | en-US)
 */
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import zhCN from '@/locales/zh-CN.json';
import enUS from '@/locales/en-US.json';

const STORAGE_KEY = 'creator_locale';
export const DEFAULT_LANG = 'zh-CN';
export type LocaleType = 'zh-CN' | 'en-US';

export function getStoredLocale(): LocaleType {
  if (typeof window === 'undefined') return DEFAULT_LANG;
  const v = localStorage.getItem(STORAGE_KEY);
  if (v === 'en-US' || v === 'zh-CN') return v;
  return DEFAULT_LANG;
}

export function setStoredLocale(locale: LocaleType): void {
  localStorage.setItem(STORAGE_KEY, locale);
  i18n.changeLanguage(locale);
}

const initialLng = typeof window !== 'undefined' ? getStoredLocale() : ('zh-CN' as LocaleType);

i18n.use(initReactI18next).init({
  resources: {
    'zh-CN': { translation: zhCN },
    'en-US': { translation: enUS },
  },
  lng: initialLng,
  fallbackLng: 'zh-CN',
  interpolation: { escapeValue: false },
});

export default i18n;

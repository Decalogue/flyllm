declare module 'umi' {
  import type { ReactNode } from 'react';

  export const history: {
    push: (path: string) => void;
    replace: (path: string) => void;
    goBack: () => void;
  };

  export function useLocation(): {
    pathname: string;
    search: string;
    hash: string;
    state: any;
  };

  export function Outlet(): ReactNode;
  export function Link(props: { to: string; children?: ReactNode; [k: string]: any }): ReactNode;
}

// Umi define 注入的全局变量
declare const API_URL: string;

// markdown-it-katex 类型声明
declare module 'markdown-it-katex' {
  import { PluginSimple } from 'markdown-it';
  const mk: PluginSimple;
  export default mk;
} 
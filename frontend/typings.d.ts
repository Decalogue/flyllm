import 'umi/typings';
import 'markdown-it';
declare global {
    const ACTION_URL: string;
    const DET_URL: string;
    /** 见 config/config.ts define，用于星图侧栏打开 Markdown 原文 */
    const STARS_DOC_BASE_URL: string;
}
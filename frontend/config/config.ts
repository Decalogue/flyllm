import { defineConfig } from 'umi';
import { routes } from './routes';
export default defineConfig({
    esbuildMinifyIIFE: true,
    plugins: ['@umijs/plugins/dist/styled-components','@umijs/plugins/dist/antd'],
    styledComponents: {},
    antd:{},
    routes: routes,
    npmClient: 'pnpm',
    define:{
        // 后端 API（对话等）。本地开发时按需改为你的服务地址
        'API_URL':'http://azj1.dc.huixingyun.com:53115',
        /** 星图「打开原文」链接前缀，例如 https://github.com/owner/flyllm/blob/main（勿尾斜杠）；留空则只显示仓库路径 */
        STARS_DOC_BASE_URL: process.env.STARS_DOC_BASE_URL || '',
    },
});

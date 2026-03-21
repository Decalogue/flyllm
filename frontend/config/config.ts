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
        // 创作 API 地址。本地开发时改为 http://localhost:5200（需先启动 creator_api）
        'API_URL':'http://azj1.dc.huixingyun.com:53115',
    },
});

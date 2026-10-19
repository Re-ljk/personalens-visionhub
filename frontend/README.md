# PersonaLens VisionHub Web

PersonaLens VisionHub 行人图文检索与可视化系统的前端界面，基于 React、TypeScript、Ant Design 和 Vite，提供描述输入、双路结果对照与图片浏览。

完整的系统介绍、部署步骤和接口说明见[项目 README](../README.md)。

## 环境

Node.js 22.12+ 与 npm。

## 开发

```bash
cp .env.example .env.local
npm install
npm run dev
```

在 `.env.local` 中设置后端地址：

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:4523/api/v1
```

## 检查与构建

```bash
npm run lint
npm run build
```

静态预览：

```bash
npx serve -s dist -l 4522 --no-clipboard
```

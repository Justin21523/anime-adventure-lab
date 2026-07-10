import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

const apiTarget = process.env.VITE_PROXY_TARGET || process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000'
const publicBase = process.env.VITE_PUBLIC_BASE || '/'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: publicBase.endsWith('/') ? publicBase : `${publicBase}/`,
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      '/api/v1': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/api/v2': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
    cors: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: true,
    chunkSizeWarningLimit: 1000,
  },
  preview: {
    port: 3000,
    strictPort: true,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
})

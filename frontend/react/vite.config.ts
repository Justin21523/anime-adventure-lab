import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api/v1': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Output directory
    outDir: 'dist',

    // Generate sourcemap for production debugging (can be disabled for smaller builds)
    sourcemap: false,

    // Rollup options
    rollupOptions: {
      output: {
        // Manual chunking for better code splitting
        manualChunks: {
          // Vendor chunk for core React libraries
          'react-vendor': ['react', 'react-dom'],

          // React Query and state management
          'query-vendor': ['@tanstack/react-query', 'zustand'],

          // UI library
          'ui-vendor': [
            '@radix-ui/react-dialog',
            '@radix-ui/react-select',
            '@radix-ui/react-toast',
            '@radix-ui/react-progress',
          ],

          // API and utilities
          'utils-vendor': ['axios', 'clsx', 'tailwind-merge'],
        },

        // Asset file naming
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name?.split('.')
          const ext = info?.[info.length - 1]
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext || '')) {
            return `assets/images/[name]-[hash][extname]`
          } else if (/woff|woff2|eot|ttf|otf/i.test(ext || '')) {
            return `assets/fonts/[name]-[hash][extname]`
          }
          return `assets/[name]-[hash][extname]`
        },

        // Chunk file naming
        chunkFileNames: 'assets/js/[name]-[hash].js',
        entryFileNames: 'assets/js/[name]-[hash].js',
      },
    },

    // Chunk size warnings
    chunkSizeWarningLimit: 1000,

    // Minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
        drop_debugger: true,
      },
    },
  },

  // Preview server configuration
  preview: {
    port: 3000,
    strictPort: true,
  },
})

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5175,
    proxy: {
      '/health':   'http://127.0.0.1:8002',
      '/metadata': 'http://127.0.0.1:8002',
      '/tchp':     'http://127.0.0.1:8002',
      '/track':    'http://127.0.0.1:8002',
      '/profile':  'http://127.0.0.1:8002',
      '/map_layer':'http://127.0.0.1:8002',
    },
  },
});

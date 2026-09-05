import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// El build va a `../static/dist`, que es de donde Flask sirve los estaticos.
// En desarrollo (`npm run dev`) las llamadas a la API y el WebSocket se
// redirigen al Flask de localhost:5000, para poder trabajar con datos reales
// del cluster sin montar nada mas.
export default defineConfig({
  plugins: [react()],
  base: '/static/dist/',
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5000',
      '/socket.io': { target: 'http://localhost:5000', ws: true },
      '/static': 'http://localhost:5000',
    },
  },
});

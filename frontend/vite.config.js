import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://localhost:5000',
      '/predict/timeseries': 'http://localhost:5000',
      '/explain': 'http://localhost:5000',
      '/history': 'http://localhost:5000',
    },
  },
})

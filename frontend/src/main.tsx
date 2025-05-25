import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import './styles/gradients.css'
import App from './App'
// import TestApp from './TestApp'

console.log('Main.tsx is executing')

const rootElement = document.getElementById('root')
if (!rootElement) {
  console.error('Root element not found')
  throw new Error('Root element not found')
}

console.log('Root element found:', rootElement)

// Create root and render app
const root = ReactDOM.createRoot(rootElement)
console.log('Root created, rendering app...')

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

console.log('App rendered')

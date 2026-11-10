import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { HOSTED } from './remote/connection'; // first: it reads a pairing link before the app reads the address
import { App } from './app/App';
import './styles/app.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// The offline copy of the app itself. Not on the computer, where it would hide fresh builds.
const onThisComputer = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
if (import.meta.env.PROD && location.protocol !== 'capacitor:' && 'serviceWorker' in navigator && (HOSTED || !onThisComputer)) {
  void navigator.serviceWorker.register('./sw.js', { scope: './' }).catch(() => undefined);
}

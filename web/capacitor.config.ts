import type { CapacitorConfig } from '@capacitor/cli';

// The iPhone / iPad app: the same web app, packaged with its files inside (npm run build:app).
const config: CapacitorConfig = {
  appId: 'io.github.dafakephoenixhk.nathanchordify',
  appName: 'Nathan Chordify',
  webDir: 'dist-app',
  ios: {
    contentInset: 'never',
    backgroundColor: '#14161c',
  },
};

export default config;

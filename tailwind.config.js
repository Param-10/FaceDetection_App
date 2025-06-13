/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neon-blue': '#00f5ff',
        'neon-purple': '#bf00ff',
        'neon-pink': '#ff0080',
        'neon-green': '#00ff88',
      },
    },
  },
  plugins: [],
} 
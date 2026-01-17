/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      borderRadius: {
        sm: '8px',
        DEFAULT: '12px',
        lg: '16px',
      },
      colors: {
        bg: {
          primary: '#15151e',
          secondary: '#1e1e2a',
          card: '#262636',
          elevated: '#2e2e40',
        },
        accent: {
          DEFAULT: '#a078ff',
          dim: '#8860f0',
          glow: '#c4a6ff',
        },
        text: {
          primary: '#fafafc',
          secondary: '#c8c8d4',
          muted: '#9898b0',
        }
      },
      boxShadow: {
        'glow': '0 0 40px -10px rgba(160, 120, 255, 0.6)',
        'glow-sm': '0 0 20px -5px rgba(160, 120, 255, 0.5)',
        'inner-glow': 'inset 0 1px 0 0 rgba(255,255,255,0.1)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E\")",
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}

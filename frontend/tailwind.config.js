/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: {
          light: '#F8F7F3',
          DEFAULT: '#EEECE5',
          dark: '#E2DFD6',
        },
        ink: {
          light: '#354340',
          DEFAULT: '#202A28',
          dark: '#131B1A',
        },
        signal: {
          light: '#CE5E47',
          DEFAULT: '#B84D38',
          dark: '#933B29',
        },
        sage: {
          light: '#BCC2AB',
          DEFAULT: '#A6AD91',
          dark: '#8B9375',
        },
        neutral: {
          light: '#A0A29A',
          DEFAULT: '#85877F',
          dark: '#696B64',
        }
      },
      fontFamily: {
        serif: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(32, 42, 40, 0.05)',
        'academic': '0 4px 20px -2px rgba(32, 42, 40, 0.08)',
      }
    },
  },
  plugins: [],
}

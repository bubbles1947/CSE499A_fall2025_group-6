/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./App.{js,jsx,ts,tsx}",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        primary: "#7C3AED",
        "primary-dark": "#6D28D9",
        "primary-light": "#8B5CF6",
        secondary: "#1E1E2E",
        dark: {
          100: "#2A2A3E",
          200: "#1E1E2E",
          300: "#15152A",
          400: "#0F0F1A",
        },
        accent: "#00D9FF",
        bubble: {
          user: "#7C3AED",
          assistant: "#2A2A3E",
        },
      },
    },
  },
  plugins: [],
};

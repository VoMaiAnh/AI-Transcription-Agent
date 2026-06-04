import { createRequire } from "node:module";

const localRequire = createRequire(import.meta.url);
const eslintRequire = createRequire(process.argv[1] ?? import.meta.url);

function requirePackage(name) {
  try {
    return localRequire(name);
  } catch {
    return eslintRequire(name);
  }
}

const importPlugin = requirePackage("eslint-plugin-import");
const jsxA11yPlugin = requirePackage("eslint-plugin-jsx-a11y");
const reactPlugin = requirePackage("eslint-plugin-react");
const reactHooksPlugin = requirePackage("eslint-plugin-react-hooks");
const tsParser = requirePackage("@typescript-eslint/parser");
const tsPlugin = requirePackage("@typescript-eslint/eslint-plugin");

const browserGlobals = {
  AudioContext: "readonly",
  Blob: "readonly",
  CanvasRenderingContext2D: "readonly",
  File: "readonly",
  FileReader: "readonly",
  FormData: "readonly",
  HTMLAudioElement: "readonly",
  HTMLDivElement: "readonly",
  HTMLInputElement: "readonly",
  HTMLTextAreaElement: "readonly",
  MediaRecorder: "readonly",
  RequestInit: "readonly",
  Response: "readonly",
  URL: "readonly",
  clearInterval: "readonly",
  clearTimeout: "readonly",
  console: "readonly",
  document: "readonly",
  fetch: "readonly",
  localStorage: "readonly",
  navigator: "readonly",
  setInterval: "readonly",
  setTimeout: "readonly",
  window: "readonly",
};

export default [
  {
    ignores: ["frontend/dist/**", "frontend/node_modules/**"],
  },
  {
    files: ["frontend/**/*.{js,jsx,ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaFeatures: { jsx: true },
        ecmaVersion: "latest",
        sourceType: "module",
      },
      globals: browserGlobals,
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
      import: importPlugin,
      "jsx-a11y": jsxA11yPlugin,
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
    },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      ...reactPlugin.configs.recommended.rules,
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "off",
      "react/react-in-jsx-scope": "off",
      "react/prop-types": "off",
      "@typescript-eslint/no-explicit-any": "off",
    },
    settings: {
      react: { version: "detect" },
    },
  },
];

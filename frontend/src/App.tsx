/**
 * Main App Component with Routing
 */
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { TranscriptionPage, TTSPage, HistoryPage, SubtitleGeneratorPage } from './pages';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<TranscriptionPage />} />
          <Route path="/subtitles" element={<SubtitleGeneratorPage />} />
          <Route path="/tts" element={<TTSPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
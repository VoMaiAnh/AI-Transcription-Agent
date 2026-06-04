import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import {
  ArchivePage,
  DashboardPage,
  DubbingStudioPage,
  LiveEditorPage,
  StatusPage,
} from "./pages";

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/editor" element={<LiveEditorPage />} />
          <Route path="/editor/:transcriptionId" element={<LiveEditorPage />} />
          <Route path="/studio" element={<DubbingStudioPage />} />
          <Route
            path="/studio/:transcriptionId"
            element={<DubbingStudioPage />}
          />
          <Route path="/archive" element={<ArchivePage />} />
          <Route path="/status" element={<StatusPage />} />
          <Route
            path="/subtitles"
            element={<Navigate to="/studio" replace />}
          />
          <Route path="/tts" element={<Navigate to="/editor" replace />} />
          <Route path="/history" element={<Navigate to="/archive" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;

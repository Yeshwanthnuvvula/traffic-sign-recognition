import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout';
import { CoverPage } from './pages/CoverPage';
import { StudioPage } from './pages/StudioPage';
import { DataExplorerPage } from './pages/DataExplorerPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { PerformancePage } from './pages/PerformancePage';
import { MethodologyPage } from './pages/MethodologyPage';
import { ReportPage } from './pages/ReportPage';

export const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<CoverPage />} />
          <Route path="studio" element={<StudioPage />} />
          <Route path="data" element={<DataExplorerPage />} />
          <Route path="experiments" element={<ExperimentsPage />} />
          <Route path="performance" element={<PerformancePage />} />
          <Route path="methodology" element={<MethodologyPage />} />
          <Route path="report" element={<ReportPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;

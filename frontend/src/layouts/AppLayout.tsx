import React from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Activity, Database, Cpu, BarChart3, Layers, BookOpen, ShieldAlert } from 'lucide-react';

export const AppLayout: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Research Cover', number: '01', icon: BookOpen },
    { path: '/studio', label: 'Recognition Studio', number: '02', icon: Activity },
    { path: '/data', label: 'Data Explorer', number: '03', icon: Database },
    { path: '/experiments', label: 'Model Experiments', number: '04', icon: Cpu },
    { path: '/performance', label: 'Model Performance', number: '05', icon: BarChart3 },
    { path: '/methodology', label: 'Methodology', number: '06', icon: Layers },
    { path: '/report', label: 'Research Report', number: '07', icon: ShieldAlert },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-paper text-ink selection:bg-signal selection:text-paper-light">
      {/* Top Academic Journal Masthead Header */}
      <header className="border-b border-ink/15 bg-paper sticky top-0 z-50 backdrop-blur-md bg-paper/95">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20 border-b border-ink/10">
            {/* Title / Brand */}
            <div className="flex items-baseline space-x-4">
              <span className="font-serif text-3xl font-bold tracking-tight text-ink">
                SIGNAL <span className="text-signal italic font-serif">/</span> VISION
              </span>
              <span className="hidden md:inline-block font-mono text-xs uppercase tracking-widest text-neutral border-l border-ink/20 pl-4">
                GTSRB Benchmark Lab • 43-Class System
              </span>
            </div>

            {/* University Project Status Pill */}
            <div className="flex items-center space-x-3 text-xs font-mono">
              <span className="inline-flex items-center px-2.5 py-1 rounded-sm bg-sage/20 text-ink border border-sage/40">
                <span className="w-1.5 h-1.5 rounded-full bg-signal mr-2 animate-pulse"></span>
                Inference Ready (MPS Active)
              </span>
            </div>
          </div>

          {/* Academic Tab Navigation */}
          <nav className="flex space-x-1 sm:space-x-4 overflow-x-auto py-2.5 scrollbar-none text-xs font-mono">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={`flex items-center px-3 py-1.5 rounded-sm transition-colors whitespace-nowrap border ${
                    isActive
                      ? 'border-signal bg-signal/10 text-signal font-semibold'
                      : 'border-transparent text-ink/70 hover:text-ink hover:border-ink/20 hover:bg-ink/5'
                  }`}
                >
                  <span className="text-neutral mr-1.5">{item.number}</span>
                  <Icon className="w-3.5 h-3.5 mr-1.5 opacity-80" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Page Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      {/* Academic Editorial Footer */}
      <footer className="border-t border-ink/15 bg-paper-dark/40 py-8 text-xs font-mono text-neutral">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <p className="font-serif italic text-sm text-ink mb-1">
              "Understanding Traffic Signs Through Machine Learning and Hyperparameter Optimization."
            </p>
            <p>German Traffic Sign Recognition Benchmark (GTSRB) • 43 Categories • 39,209 Train / 12,630 Test</p>
          </div>
          <div className="text-right flex items-center space-x-6">
            <span>Python 3.12 • PyTorch MPS • FastAPI • React</span>
            <span className="border-l border-ink/20 pl-4">University CS Thesis Project</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

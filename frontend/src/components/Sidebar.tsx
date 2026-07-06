import React from 'react';
import { LayoutDashboard, MessageSquare, Shield, Cpu, RefreshCw, Trash2, Download, Users } from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onClear: () => void;
  isClearing: boolean;
  vectorDbInfo: string;
  onExportPdf?: () => void;
  isExportingPdf?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  onClear,
  isClearing,
  vectorDbInfo,
  onExportPdf,
  isExportingPdf = false,
}) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'debate', label: 'Medical Board Debate', icon: Users },
    { id: 'coach', label: 'AI Health Coach', icon: MessageSquare },
    { id: 'biomarkers', label: 'Biomarker Vault', icon: Shield },
    { id: 'diagnostics', label: 'Agent Diagnostics', icon: Cpu },
  ];

  return (
    <div className="sidebar-panel font-sans flex flex-col justify-between h-full p-6 text-white select-none">
      <div>
        {/* Logo and Brand in Lora Serif */}
        <div className="flex items-center gap-3 mb-8 px-2">
          <div className="w-8 h-8 rounded-lg bg-[#C8A97A] flex items-center justify-center font-bold text-black text-lg select-none">
            N
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-[#F7F5F0] font-serif">
              NutriMind AI
            </h1>
            <p className="text-[10px] uppercase tracking-widest text-[#9E9990] font-semibold">
              Clinical Agentic Core
            </p>
          </div>
        </div>

        {/* Navigation List */}
        <nav className="flex flex-col gap-1">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  width: '100%',
                  padding: '12px 16px',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? '#F7F5F0' : '#9E9990',
                  background: isActive ? '#252523' : 'transparent',
                  border: 'none',
                  borderLeft: isActive ? '3px solid #C8A97A' : '3px solid transparent',
                  paddingLeft: isActive ? '13px' : '16px', // Compensate border width
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.2s ease',
                }}
                className="hover:text-white hover:bg-[#252523] group"
              >
                <Icon
                  size={18}
                  style={{
                     color: isActive ? '#C8A97A' : '#6A6660',
                     transition: 'color 0.2s ease',
                  }}
                  className="group-hover:text-[#C8A97A]"
                />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info & Reset Actions */}
      <div className="flex flex-col gap-4 border-t border-[#252523] pt-6">
        {/* Dynamic Vector DB & Model Display */}
        <div className="px-2">
          <div className="text-[10px] uppercase font-semibold text-[#6A6660] tracking-wider mb-1">
            Active Vector Core
          </div>
          <div className="text-xs text-[#9E9990] bg-[#141413] border border-[#252523] p-2.5 rounded-md font-mono select-text break-words">
            {vectorDbInfo || 'ChromaDB Core (HNSW)'}
          </div>
        </div>

        {/* Action Buttons Wrapper */}
        <div className="flex flex-col gap-2">
          {/* Export PDF Report Button */}
          <button
            onClick={onExportPdf}
            disabled={isExportingPdf}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              width: '100%',
              padding: '10px',
              background: 'transparent',
              border: '0.5px solid #C8A97A',
              color: '#C8A97A',
              fontSize: '0.85rem',
              fontWeight: 600,
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
            className="hover:bg-[#C8A97A]/10 hover:border-[#C8A97A]"
          >
            {isExportingPdf ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <Download size={14} />
            )}
            Export Clinical PDF
          </button>

          {/* Clear Database button */}
          <button
            onClick={onClear}
            disabled={isClearing}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              width: '100%',
              padding: '10px',
              background: 'transparent',
              border: '0.5px solid #6A6660',
              color: '#B25E5E',
              fontSize: '0.85rem',
              fontWeight: 600,
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
            className="hover:bg-[#B25E5E]/10 hover:border-[#B25E5E]"
          >
            {isClearing ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <Trash2 size={14} />
            )}
            Clear Clinical Index
          </button>
        </div>
      </div>
    </div>
  );
};

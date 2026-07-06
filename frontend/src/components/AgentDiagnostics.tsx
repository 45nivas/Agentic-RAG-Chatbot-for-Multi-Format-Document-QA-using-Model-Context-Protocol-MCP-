import React, { useState } from 'react';
import type { AgentTrace } from '../types';
import { Cpu, Terminal, CheckCircle2, AlertTriangle, Play, Eye, EyeOff } from 'lucide-react';

interface AgentDiagnosticsProps {
  traces: AgentTrace[];
  vectorDbInfo: string;
}

export const AgentDiagnostics: React.FC<AgentDiagnosticsProps> = ({ traces, vectorDbInfo }) => {
  const [statusFilter, setStatusFilter] = useState<'all' | 'success' | 'error'>('all');
  const [expandedTraceIdx, setExpandedTraceIdx] = useState<number | null>(null);

  // Baseline mock traces if none exists yet to showcase the clinical pipeline
  const defaultTraces: AgentTrace[] = [
    {
      sender: "IngestionAgent",
      receiver: "ClinicalAnalyzerAgent",
      type: "DOCUMENT_INGEST",
      payload: { files: ["biomarker_report_patient_5.pdf"], characters_extracted: 14250, status: "SUCCESS" }
    },
    {
      sender: "ClinicalAnalyzerAgent",
      receiver: "CoordinatorAgent",
      type: "CLINICAL_ANALYSIS",
      payload: { age: 34, weight_kg: 82, biomarkers_extracted: ["Glucose", "LDL Cholesterol", "Vitamin D"], gender: "Female" }
    },
    {
      sender: "WebResearchAgent",
      receiver: "NutriPlannerAgent",
      type: "RESEARCH_NOTE",
      payload: { search_source: "PubMed Database & AHA Guidelines", citations: 12, guidance: "Reduce sodium, increase magnesium/potassium, target whole foods." }
    },
    {
      sender: "NutriPlannerAgent",
      receiver: "SafetyAuditorAgent",
      type: "MEAL_PLAN",
      payload: { calorie_target: 1950, macros: { p: 135, c: 180, f: 65 }, meals_calculated: ["Breakfast", "Lunch", "Dinner", "Snack"] }
    },
    {
      sender: "SafetyAuditorAgent",
      receiver: "CoordinatorAgent",
      type: "SAFETY_AUDIT_RESULT",
      payload: { is_cleared: true, allergies_checked: ["gluten", "peanuts"], corrections_made: ["Replaced whole-wheat with GF sourdough"] }
    },
    {
      sender: "LLMResponseAgent",
      receiver: "UI",
      type: "RESPONSE",
      payload: { response_length: 512, model_used: "gemini-2.5-flash", safety_audit: "CLEARED" }
    }
  ];

  const activeTraces = traces.length > 0 ? traces : defaultTraces;

  // Filter traces
  const filteredTraces = activeTraces.filter(t => {
    if (statusFilter === 'all') return true;
    const isErr = t.type.toLowerCase().includes('error') || t.payload?.status === 'ERROR' || t.payload?.success === false;
    if (statusFilter === 'error') return isErr;
    return !isErr;
  });

  const toggleExpandTrace = (idx: number) => {
    setExpandedTraceIdx(expandedTraceIdx === idx ? null : idx);
  };

  const getTraceIcon = (type: string, payload: any) => {
    const isErr = type.toLowerCase().includes('error') || payload?.status === 'ERROR' || payload?.success === false;
    if (isErr) return <AlertTriangle size={16} style={{ color: '#B25E5E' }} />;
    return <CheckCircle2 size={16} style={{ color: '#4D7C5D' }} />;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', padding: '24px' }}>
      
      {/* Telemetry Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        
        {/* Core database info */}
        <div className="clinical-card" style={{ display: 'flex', alignItems: 'center', gap: '14px', margin: 0, padding: '16px' }}>
          <div style={{ backgroundColor: '#F4EFE6', padding: '10px', borderRadius: '8px', color: '#A58452' }}>
            <Cpu size={20} />
          </div>
          <div>
            <div style={{ fontSize: '0.65rem', fontWeight: 700, color: '#9E9990', textTransform: 'uppercase', letterSpacing: '0.05em' }}>VECTOR DATABASE</div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1A1A18' }}>ChromaDB + Cosine HNSW</div>
          </div>
        </div>

        {/* Model Spec */}
        <div className="clinical-card" style={{ display: 'flex', alignItems: 'center', gap: '14px', margin: 0, padding: '16px' }}>
          <div style={{ backgroundColor: '#F4EFE6', padding: '10px', borderRadius: '8px', color: '#A58452' }}>
            <Terminal size={20} />
          </div>
          <div>
            <div style={{ fontSize: '0.65rem', fontWeight: 700, color: '#9E9990', textTransform: 'uppercase', letterSpacing: '0.05em' }}>EMBEDDING MODEL</div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1A1A18', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '180px' }}>
              {vectorDbInfo.includes('Model:') ? vectorDbInfo.split('Model:')[1].trim() : 'PubMedBERT clinical'}
            </div>
          </div>
        </div>

        {/* Dimensionality stats */}
        <div className="clinical-card" style={{ display: 'flex', alignItems: 'center', gap: '14px', margin: 0, padding: '16px' }}>
          <div style={{ backgroundColor: '#F4EFE6', padding: '10px', borderRadius: '8px', color: '#A58452' }}>
            <Play size={20} />
          </div>
          <div>
            <div style={{ fontSize: '0.65rem', fontWeight: 700, color: '#9E9990', textTransform: 'uppercase', letterSpacing: '0.05em' }}>DIMENSIONS & CACHE</div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1A1A18' }}>
              {vectorDbInfo.toLowerCase().includes('pubmedbert') ? '768-D' : '384-D'} (LRU Cache Active)
            </div>
          </div>
        </div>

      </div>

      {/* Interactive Trace Log Panel */}
      <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        
        {/* Header and filters */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', borderBottom: '0.5px solid #dedad4', paddingBottom: '12px' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#1A1A18' }}>Multi-Agent Execution Pipeline Trace</h3>
            <p style={{ fontSize: '0.75rem', color: '#6A6660' }}>Direct step-by-step logs from the Clinical Agentic Core</p>
          </div>

          {/* Filters buttons */}
          <div style={{ display: 'flex', border: '0.5px solid #dedad4', borderRadius: '6px', overflow: 'hidden' }}>
            {(['all', 'success', 'error'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setStatusFilter(mode)}
                style={{
                  background: statusFilter === mode ? '#1A1A18' : 'transparent',
                  color: statusFilter === mode ? '#FFFFFF' : '#6A6660',
                  border: 'none',
                  padding: '6px 12px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                  transition: 'all 0.15s ease',
                }}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        {/* Trace Table */}
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid #dedad4', color: '#6A6660', fontWeight: 600 }}>
                <th style={{ padding: '12px 8px' }}>Pipeline Role</th>
                <th style={{ padding: '12px 8px' }}>Operation</th>
                <th style={{ padding: '12px 8px' }}>Type</th>
                <th style={{ padding: '12px 8px' }}>Target Receiver</th>
                <th style={{ padding: '12px 8px', textAlign: 'center' }}>Details</th>
              </tr>
            </thead>
            <tbody>
              {filteredTraces.map((trace, idx) => {
                const isExpanded = expandedTraceIdx === idx;
                const senderName = trace.sender.replace('Agent', '');
                const receiverName = trace.receiver.replace('Agent', '');
                
                return (
                  <React.Fragment key={idx}>
                    {/* Row header */}
                    <tr style={{
                      borderBottom: '0.5px solid #F4EFE6',
                      backgroundColor: isExpanded ? '#F7F5F0' : 'transparent',
                      transition: 'background-color 0.2s ease'
                    }}>
                      <td style={{ padding: '12px 8px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', color: '#1A1A18' }}>
                        {getTraceIcon(trace.type, trace.payload)}
                        {senderName}
                      </td>
                      <td style={{ padding: '12px 8px', color: '#6A6660' }}>
                        <span style={{
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          backgroundColor: '#EBE8E3',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          color: '#1A1A18',
                          fontWeight: 600
                        }}>
                          {trace.type}
                        </span>
                      </td>
                      <td style={{ padding: '12px 8px', color: '#6A6660' }}>
                        {trace.type.includes('ERROR') ? 'Pipeline Exception' : 'Agent Protocol'}
                      </td>
                      <td style={{ padding: '12px 8px', color: '#6A6660', fontWeight: 500 }}>
                        {receiverName}
                      </td>
                      <td style={{ padding: '12px 8px', textAlign: 'center' }}>
                        <button
                          onClick={() => toggleExpandTrace(idx)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#C8A97A',
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontWeight: 600
                          }}
                        >
                          {isExpanded ? <EyeOff size={14} /> : <Eye size={14} />}
                          {isExpanded ? 'Hide' : 'Inspect'}
                        </button>
                      </td>
                    </tr>

                    {/* Expandable JSON Drawer */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={5} style={{ padding: '16px', backgroundColor: '#F4EFE6', borderBottom: '0.5px solid #dedad4' }}>
                          <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '8px',
                          }}>
                            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#A58452', textTransform: 'uppercase' }}>
                              MCP Payload Tracing Logs
                            </div>
                            <pre style={{
                              margin: 0,
                              backgroundColor: '#1E1E1E',
                              color: '#A9D18E',
                              padding: '14px',
                              borderRadius: '6px',
                              fontFamily: 'monospace',
                              fontSize: '0.75rem',
                              overflowX: 'auto',
                              boxShadow: 'inset 0 1px 4px rgba(0,0,0,0.2)',
                              maxHeight: '260px',
                            }}>
                              {JSON.stringify(trace.payload, null, 2)}
                            </pre>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}

              {filteredTraces.length === 0 && (
                <tr>
                  <td colSpan={5} style={{ padding: '24px', textAlign: 'center', color: '#9E9990' }}>
                    No execution traces found matching status filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

      </div>

    </div>
  );
};

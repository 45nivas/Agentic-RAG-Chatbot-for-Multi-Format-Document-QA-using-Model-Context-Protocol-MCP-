import React, { useState, useEffect, useRef } from 'react';
import type { AgentTrace } from '../types';
import { BookOpen, ShieldAlert, Award, Dumbbell, Apple, Heart, User, CheckCircle, AlertTriangle, Play, HelpCircle, Users } from 'lucide-react';

interface BoardDebateProps {
  mcpTraces: AgentTrace[];
}

export const BoardDebate: React.FC<BoardDebateProps> = ({ mcpTraces }) => {
  const [visibleCount, setVisibleCount] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const containerEndRef = useRef<HTMLDivElement | null>(null);

  // Restart animation when new traces arrive
  useEffect(() => {
    if (mcpTraces.length > 0) {
      setVisibleCount(1);
      setIsPlaying(true);
    } else {
      setVisibleCount(0);
      setIsPlaying(false);
    }
  }, [mcpTraces]);

  // Handle debate step timer
  useEffect(() => {
    if (isPlaying && visibleCount < mcpTraces.length) {
      timerRef.current = setTimeout(() => {
        setVisibleCount(prev => prev + 1);
      }, 2000); // 2 seconds pacing per agent speech
    } else {
      setIsPlaying(false);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, visibleCount, mcpTraces]);

  // Scroll debate container to bottom on new visible logs
  useEffect(() => {
    containerEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [visibleCount]);

  const handleRestart = () => {
    setVisibleCount(1);
    setIsPlaying(true);
  };

  const getAgentMeta = (sender: string) => {
    switch (sender) {
      case 'IngestionAgent':
        return {
          title: 'Document Ingestion Parser',
          icon: BookOpen,
          color: '#64748B', // slate
          bg: '#F8FAFC',
          border: '1px solid #CBD5E1',
        };
      case 'ClinicalAnalyzerAgent':
        return {
          title: 'Biomarker Extraction Analyst',
          icon: ShieldAlert,
          color: '#6366F1', // indigo
          bg: '#EEF2FF',
          border: '1px solid #C7D2FE',
        };
      case 'WebResearchAgent':
        return {
          title: 'Clinical Research Librarian (PubMed)',
          icon: BookOpen,
          color: '#8B5CF6', // violet
          bg: '#F5F3FF',
          border: '1px solid #DDD6FE',
        };
      case 'BioAgeCalculatorAgent':
        return {
          title: 'Lead Gerontologist & Longevity Specialist',
          icon: Heart,
          color: '#10B981', // emerald
          bg: '#ECFDF5',
          border: '1px solid #A7F3D0',
        };
      case 'ClinicalKinesiologyAgent':
        return {
          title: 'Sports Physiologist & Kinesiologist',
          icon: Dumbbell,
          color: '#3B82F6', // blue
          bg: '#EFF6FF',
          border: '1px solid #BFDBFE',
        };
      case 'NutriPlannerAgent':
        return {
          title: 'Clinical Sports Nutritionist',
          icon: Apple,
          color: '#14B8A6', // teal
          bg: '#F0FDFA',
          border: '1px solid #99F6E4',
        };
      case 'SafetyAuditorAgent':
        return {
          title: 'Chief Safety Compliance Officer',
          icon: ShieldAlert,
          color: '#EF4444', // red
          bg: '#FEF2F2',
          border: '2px solid #FCA5A5',
        };
      case 'ClinicalCritiqueAgent':
        return {
          title: 'President of the Medical Board',
          icon: Award,
          color: '#D97706', // gold
          bg: '#FFFBEB',
          border: '2px solid #FCD34D',
        };
      case 'LLMResponseAgent':
        return {
          title: 'Patient Health Communicator',
          icon: User,
          color: '#1C1E21', // ink dark
          bg: '#F9FAF9',
          border: '1px solid #252523',
        };
      default:
        return {
          title: 'Clinical Advisor',
          icon: HelpCircle,
          color: '#6A6660',
          bg: '#FFFFFF',
          border: '1px solid #E5E7EB',
        };
    }
  };

  const renderPayload = (sender: string, payload: any) => {
    if (!payload) return null;

    try {
      if (sender === 'BioAgeCalculatorAgent' && payload.bio_age_results) {
        const res = payload.bio_age_results;
        return (
          <div className="mt-3 bg-white p-3 rounded-md border border-[#E5E7EB] font-sans text-xs flex flex-wrap gap-4">
            <div><b>Chronological:</b> {res.chronological_age} yrs</div>
            <div><b>Biological:</b> {res.biological_age} yrs</div>
            <div><b>Longevity Index:</b> {res.longevity_score}%</div>
            <div className="w-full"><b>Pathway Cellular Target:</b> {res.pathway_focus}</div>
          </div>
        );
      }

      if (sender === 'ClinicalKinesiologyAgent' && payload.training_plan) {
        const res = payload.training_plan;
        return (
          <div className="mt-3 bg-white p-3 rounded-md border border-[#E5E7EB] font-sans text-xs">
            <div className="mb-2"><b>Split:</b> {res.weekly_split}</div>
            <div className="text-[10px] uppercase font-semibold text-[#888888] tracking-wider mb-1">Prescribed Workouts:</div>
            <ul className="list-disc pl-4 space-y-1">
              {res.exercises?.slice(0, 3).map((ex: any, idx: number) => (
                <li key={idx}><b>{ex.name}</b> - {ex.sets} sets x {ex.reps} reps ({ex.intensity})</li>
              ))}
            </ul>
          </div>
        );
      }

      if (sender === 'NutriPlannerAgent' && payload.targets) {
        const res = payload.targets;
        return (
          <div className="mt-3 bg-white p-3 rounded-md border border-[#E5E7EB] font-sans text-xs flex gap-4">
            <div><b>Calorie Target:</b> {res.calories} kcal</div>
            <div><b>Macros:</b> P: {res.protein}g | C: {res.carbs}g | F: {res.fats}g</div>
          </div>
        );
      }

      if (sender === 'SafetyAuditorAgent') {
        const hasCorrections = payload.corrections_made && payload.corrections_made.length > 0 && !payload.corrections_made[0].includes("Standard safety");
        return (
          <div className="mt-3 font-sans">
            {hasCorrections ? (
              <div style={{ backgroundColor: '#FEE2E2', border: '1px solid #EF4444', color: '#991B1B' }} className="p-3 rounded-md text-xs mb-2 flex items-start gap-2.5">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                <div>
                  <div className="font-semibold text-red-800 uppercase tracking-wider text-[10px] mb-0.5">COMPLIANCE CRITICAL OVERRIDE ENFORCED</div>
                  <ul className="list-disc pl-4 space-y-1 mt-1">
                    {payload.corrections_made.map((corr: string, idx: number) => (
                      <li key={idx} className="font-medium">{corr}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div style={{ backgroundColor: '#D1FAE5', border: '1px solid #10B981', color: '#065F46' }} className="p-2.5 rounded-md text-xs mb-2 flex items-center gap-2">
                <CheckCircle size={14} />
                <span className="font-semibold uppercase tracking-wider text-[9px]">Allergen & Joint Safety Audits Cleared</span>
              </div>
            )}
            <div className="text-[10px] text-[#6A6660] italic mt-1">{payload.audit_report}</div>
          </div>
        );
      }

      if (sender === 'ClinicalCritiqueAgent' && payload.critique) {
        const res = payload.critique;
        return (
          <div className="mt-3 bg-white p-3 rounded-md border border-[#E5E7EB] font-sans text-xs">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] uppercase font-bold text-[#888888]">Review Board Verdict:</span>
              <span style={{ backgroundColor: '#FFFBEB', color: '#D97706', border: '1px solid #FCD34D' }} className="px-2 py-0.5 rounded font-bold text-sm">{res.clinical_grade}</span>
            </div>
            <div><b>Pathway Mechanics:</b> {res.mechanics_explanation?.substring(0, 180)}...</div>
          </div>
        );
      }

      // Default string payload formatting
      if (typeof payload === 'string') {
        return <p className="mt-2 text-xs leading-relaxed text-[#4B5563] break-words whitespace-pre-wrap">{payload}</p>;
      } else if (payload.research_note) {
        return <p className="mt-2 text-xs leading-relaxed text-[#4B5563] break-words whitespace-pre-wrap">{payload.research_note.substring(0, 250)}...</p>;
      } else if (payload.answer) {
        return <p className="mt-2 text-xs leading-relaxed text-[#1F2937] font-semibold break-words whitespace-pre-wrap">{payload.answer}</p>;
      }
    } catch (e) {
      console.warn("Failed rendering payload trace: ", e);
    }
    
    return null;
  };

  const visibleTraces = mcpTraces.slice(0, visibleCount);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, height: '100%' }}>
      {/* Control Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 24px',
        backgroundColor: '#FFFFFF',
        borderBottom: '1px solid #E5E7EB',
      }}
      className="select-none"
      >
        <div>
          <h3 className="font-serif text-base font-bold text-[#1A1A18]">
            Medical Advisory Board Dialogue Session
          </h3>
          <p className="text-[10px] text-[#6A6660] font-sans mt-0.5">
            Real-time visual trace tracking planning, clinical guidelines reasoning, safety audits, and peer critiques.
          </p>
        </div>

        {mcpTraces.length > 0 && (
          <button
            onClick={handleRestart}
            disabled={isPlaying}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: isPlaying ? '#E5E7EB' : '#1A1A18',
              color: isPlaying ? '#888888' : '#FFFFFF',
              border: 'none',
              padding: '6px 14px',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: isPlaying ? 'default' : 'pointer',
              transition: 'background-color 0.2s'
            }}
            className={isPlaying ? '' : 'hover:bg-[#C8A97A]'}
          >
            <Play size={12} />
            Replay Discussion
          </button>
        )}
      </div>

      {/* Main Debate Thread */}
      <div style={{
        flex: 1,
        padding: '24px',
        overflowY: 'auto',
        backgroundColor: '#F7F5F0', // warm clinical white
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        {mcpTraces.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            color: '#9E9990',
            textAlign: 'center'
          }}
          className="select-none font-sans"
          >
            <Users size={48} className="stroke-1 text-[#C8A97A] mb-4" />
            <h4 className="font-bold text-sm text-[#1A1A18]">Advisory Board Chamber is Idle</h4>
            <p className="text-xs text-[#6A6660] max-w-sm mt-1">
              Upload your clinical lab report or ask a health question in the AI Health Coach tab. Your multi-agent board will convene here to debate your plan!
            </p>
          </div>
        ) : (
          <div style={{ maxWidth: '780px', width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {visibleTraces.map((trace, index) => {
              const meta = getAgentMeta(trace.sender);
              const Icon = meta.icon;
              return (
                <div
                  key={index}
                  style={{
                    backgroundColor: meta.bg,
                    border: meta.border,
                    borderRadius: '8px',
                    padding: '16px',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03)',
                    display: 'flex',
                    gap: '14px',
                    animation: 'fadeIn 0.4s ease-out forwards',
                  }}
                  className="font-sans"
                >
                  {/* Badge Icon */}
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor: '#FFFFFF',
                    border: `1px solid ${meta.color}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: meta.color,
                    boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                  }}
                  className="shrink-0"
                  >
                    <Icon size={18} />
                  </div>

                  {/* Speech Bubble */}
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #E5E7EB', paddingBottom: '4px', marginBottom: '8px' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 700, color: meta.color }}>
                        {meta.title}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: '#888888', fontWeight: 500 }}>
                        {trace.sender === 'LLMResponseAgent' ? 'Decision Communicated' : 'Deliberation Step'}
                      </span>
                    </div>

                    {renderPayload(trace.sender, trace.payload)}
                  </div>
                </div>
              );
            })}

            {/* Simulated Loading Indicator */}
            {isPlaying && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '14px 20px',
                backgroundColor: '#FFFFFF',
                border: '1.5px dashed #CBD5E1',
                borderRadius: '8px',
                maxWidth: '240px',
                animation: 'pulse 1.5s infinite ease-in-out'
              }}
              className="font-sans select-none"
              >
                <div className="w-2 h-2 bg-[#C8A97A] rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                <div className="w-2 h-2 bg-[#C8A97A] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-[#C8A97A] rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                <span className="text-xs font-semibold text-[#6A6660]">Advisory Board is debating...</span>
              </div>
            )}

            <div ref={containerEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

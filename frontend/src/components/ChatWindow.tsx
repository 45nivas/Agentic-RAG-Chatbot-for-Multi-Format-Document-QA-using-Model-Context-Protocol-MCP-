import React, { useState, useRef, useEffect } from 'react';
import type { ChatMessage } from '../types';
import { Send, Sparkles, MessageSquare, Cpu, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';

interface ChatWindowProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isSending: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  onSendMessage,
  isSending,
}) => {
  const [inputText, setInputText] = useState('');
  const [expandedTraceIdx, setExpandedTraceIdx] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isSending) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const toggleTrace = (idx: number) => {
    setExpandedTraceIdx(expandedTraceIdx === idx ? null : idx);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', backgroundColor: '#F7F5F0' }}>
      
      {/* Chat Messages Panel */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {messages.length === 0 ? (
          /* Empty Chat state */
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#9E9990',
            textAlign: 'center',
            gap: '12px',
            maxWidth: '400px',
            margin: 'auto'
          }}>
            <div style={{ backgroundColor: '#F4EFE6', padding: '14px', borderRadius: '50%', color: '#C8A97A' }}>
              <MessageSquare size={32} />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: '#1A1A18', fontFamily: 'Lora, serif' }}>
              NutriMind Clinical Assistant
            </h3>
            <p style={{ fontSize: '0.85rem', color: '#6A6660', lineHeight: '1.45' }}>
              Ask clinical fitness questions or discuss active diet programs. The Plan-Reason-Audit agent core will critique suggestions safely.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center', marginTop: '10px' }}>
              {["What foods should I avoid with high LDL?", "Optimize my protein targets", "Suggest a pre-workout breakfast"].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => onSendMessage(suggestion)}
                  style={{
                    fontSize: '0.75rem',
                    color: '#6A6660',
                    backgroundColor: '#FFFFFF',
                    border: '0.5px solid #dedad4',
                    padding: '6px 12px',
                    borderRadius: '20px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                  className="hover:border-[#C8A97A] hover:bg-[#F4EFE6]/30"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Message List */
          messages.map((msg, idx) => {
            const isUser = msg.sender === 'user';
            
            return (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: isUser ? 'flex-end' : 'flex-start',
                  width: '100%',
                }}
              >
                {/* Message Bubble */}
                <div
                  style={{
                    maxWidth: '80%',
                    backgroundColor: isUser ? '#1A1A18' : '#FFFFFF',
                    color: isUser ? '#FFFFFF' : '#1A1A18',
                    border: '0.5px solid #dedad4',
                    borderColor: isUser ? '#1A1A18' : '#dedad4',
                    borderRadius: '12px',
                    borderTopRightRadius: isUser ? '2px' : '12px',
                    borderTopLeftRadius: isUser ? '12px' : '2px',
                    padding: '14px 18px',
                    fontSize: '0.92rem',
                    lineHeight: '1.5',
                    boxShadow: isUser ? 'none' : 'var(--shadow-card)',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {msg.text}
                </div>

                {/* Safety warnings or corrections indicator (if safety checks made corrections) */}
                {!isUser && msg.corrections && msg.corrections.length > 0 && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '0.75rem',
                    color: '#B25E5E',
                    backgroundColor: '#F9EFEF',
                    border: '0.5px solid #B25E5E40',
                    padding: '4px 12px',
                    borderRadius: '4px',
                    marginTop: '6px',
                    fontWeight: 600
                  }}>
                    <AlertCircle size={12} />
                    Safety Auditor amended menu suggestions: {msg.corrections.join(', ')}
                  </div>
                )}

                {/* Collapsible Chain-of-Thought (CoT) Logs */}
                {!isUser && msg.mcpTrace && msg.mcpTrace.length > 0 && (
                  <div style={{ width: '100%', maxWidth: '80%', marginTop: '8px' }}>
                    <button
                      onClick={() => toggleTrace(idx)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '0.75rem',
                        color: '#6A6660',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        fontWeight: 600,
                        padding: '2px 4px',
                      }}
                      className="hover:text-[#C8A97A]"
                    >
                      <Cpu size={12} />
                      {expandedTraceIdx === idx ? 'Hide Multi-Agent reasoning chain' : 'Inspect multi-agent reasoning chain'}
                      {expandedTraceIdx === idx ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>

                    {expandedTraceIdx === idx && (
                      <div style={{
                        marginTop: '6px',
                        border: '0.5px solid #dedad4',
                        borderRadius: '6px',
                        backgroundColor: '#FFFFFF',
                        overflow: 'hidden',
                        fontSize: '0.8rem',
                      }}>
                        <div style={{ padding: '8px 12px', backgroundColor: '#F7F5F0', borderBottom: '0.5px solid #dedad4', fontWeight: 700, color: '#A58452', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Trace Logs (Clinical Orchestrator)
                        </div>
                        <div style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '180px', overflowY: 'auto' }}>
                          {msg.mcpTrace.map((trace, tIdx) => (
                            <div key={tIdx} style={{ display: 'flex', gap: '6px', borderBottom: tIdx < msg.mcpTrace!.length - 1 ? '0.5px solid #F4EFE6' : 'none', paddingBottom: '6px' }}>
                              <span style={{ fontWeight: 700, color: '#1A1A18', whiteSpace: 'nowrap' }}>{trace.sender.replace('Agent', '')}:</span>
                              <span style={{ color: '#6A6660', fontStyle: 'italic' }}>
                                {trace.type === 'RESPONSE' ? 'Returned response payload' : 
                                 trace.type === 'CLINICAL_ANALYSIS' ? 'Extracted biomarkers profile' :
                                 trace.type === 'RESEARCH_NOTE' ? 'Synthesized PubMed clinical references' :
                                 trace.type === 'MEAL_PLAN' ? 'Computed calorie macro targets & menu structure' :
                                 trace.type === 'SAFETY_AUDIT_RESULT' ? `Allergen critique cleared: ${trace.payload.is_cleared ? 'Yes' : 'No'}` :
                                 `Executed protocol ${trace.type}`}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Timestamp */}
                <span style={{
                  fontSize: '0.7rem',
                  color: '#9E9990',
                  marginTop: '4px',
                  padding: '0 4px',
                  fontWeight: 500
                }}>
                  {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            );
          })
        )}

        {/* Loading Spinner */}
        {isSending && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', alignSelf: 'flex-start', backgroundColor: '#FFFFFF', border: '0.5px solid #dedad4', padding: '12px 18px', borderRadius: '12px', borderTopLeftRadius: '2px', boxShadow: 'var(--shadow-card)' }}>
            <Sparkles size={16} className="animate-spin text-[#C8A97A]" style={{ animation: 'spin-slow 2s linear infinite' }} />
            <span style={{ fontSize: '0.85rem', color: '#6A6660', fontWeight: 500 }}>
              Plan-Reason-Audit loop executing...
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input Message Form Panel */}
      <form onSubmit={handleSubmit} style={{
        padding: '16px 24px',
        backgroundColor: '#FFFFFF',
        borderTop: '0.5px solid #dedad4',
        display: 'flex',
        gap: '12px',
        alignItems: 'center'
      }}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask clinical diet planner or coach details..."
          disabled={isSending}
          style={{
            flex: 1,
            backgroundColor: '#F7F5F0',
            border: '0.5px solid #dedad4',
            borderRadius: '6px',
            padding: '12px 16px',
            fontSize: '0.9rem',
            color: '#1A1A18',
            fontFamily: 'var(--font-sans)',
            outline: 'none',
            transition: 'border-color 0.2s ease',
          }}
          className="focus:border-[#C8A97A]"
        />
        <button
          type="submit"
          disabled={!inputText.trim() || isSending}
          style={{
            backgroundColor: '#1A1A18',
            border: 'none',
            color: '#FFFFFF',
            width: '42px',
            height: '42px',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          className="hover:bg-[#C8A97A] disabled:opacity-50 disabled:bg-[#1A1A18] disabled:cursor-not-allowed"
        >
          <Send size={16} />
        </button>
      </form>

    </div>
  );
};

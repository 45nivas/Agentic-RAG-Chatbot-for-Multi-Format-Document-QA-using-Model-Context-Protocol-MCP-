import { useState } from 'react';
import type { TrainingPlan } from '../types';
import { Dumbbell, ShieldAlert, ChevronDown, ChevronUp, CheckSquare, Square, Upload } from 'lucide-react';

interface WorkoutProgramProps {
  trainingPlan: TrainingPlan | null;
}

export function WorkoutProgram({ trainingPlan }: WorkoutProgramProps) {
  const isRealData = trainingPlan !== null;

  // Safe default sports-science clinical routine fallback
  const plan: TrainingPlan = trainingPlan || {
    weekly_split: "3-Day Metabolic Strength Conditioning + 2-Day Zone 2 Aerobic Conditioning",
    exercises: [
      {
        name: "Controlled Dumbbell Goblet Squat",
        sets: "3",
        reps: "12",
        intensity: "RPE 7 (Moderate Intensity)",
        instructions: "Hold a single heavy dumbbell vertically at your collarbone. Slowly lower into a deep squat over a 3-second cadence, keep knees tracking outwards over toes, and return to standing while exhaling."
      },
      {
        name: "Neutral-Grip Dumbbell Floor Press",
        sets: "3",
        reps: "10-12",
        intensity: "RPE 8 (Controlled Strength)",
        instructions: "Lie flat on your back on the floor, holding dumbbells with palms facing each other. Lower arms until triceps brush the floor, pause briefly to disengage momentum, then press dynamically upward."
      },
      {
        name: "Steady-State Zone 2 Cardio (Biking or Row)",
        sets: "1",
        reps: "30-40 mins",
        intensity: "HR 125-135 BPM (Zone 2)",
        instructions: "Maintain a steady, conversation-level pace on a stationary bike or rowing machine. This target aerobic threshold encourages cellular lipid clearance and glycogen storage optimization."
      },
      {
        name: "Static bird-dog Core Holds",
        sets: "3",
        reps: "10 per side",
        intensity: "RPE 6 (Stability Focus)",
        instructions: "Position on all fours. Extend opposite arm and leg horizontally until parallel to the floor. Hold for 3 seconds, focusing on locking in lumbar spine stability before switching sides."
      }
    ],
    safety_precautions: [
      "Ensure controlled, continuous breathing during heavy movements. Avoid the Valsalva maneuver (holding breath) to prevent transient cardiovascular arterial spikes.",
      "Integrate an 8-minute progressive warm-up (e.g. arm swings, bodyweight hinges) before starting loaded sets to encourage synovial joint lubrication.",
      "If any dizziness, sudden heart rate spikes, or localized articular pain occurs, halt exercise immediately and check blood glucose markers."
    ]
  };

  const [expandedExercise, setExpandedExercise] = useState<number | null>(null);
  const [checkedExercises, setCheckedExercises] = useState<{ [key: number]: boolean }>({});

  const toggleExpand = (idx: number) => {
    setExpandedExercise(prev => (prev === idx ? null : idx));
  };

  const toggleChecked = (idx: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setCheckedExercises(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  return (
    <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '20px', position: 'relative', overflow: 'hidden' }}>
      {!isRealData && (
        <div style={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          zIndex: 10,
        }}>
          <span className="stat-pill accent-gold" style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            Sample Preview
          </span>
        </div>
      )}

      {/* Header section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '0.5px solid #dedad4', paddingBottom: '10px' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1A1A18' }}>Clinical Workout Prescription</h3>
          <p style={{ fontSize: '0.75rem', color: '#6A6660' }}>Custom sports-science program & split</p>
        </div>
        <div style={{ backgroundColor: '#1A1A18', color: '#C8A97A', padding: '6px', borderRadius: '50%' }}>
          <Dumbbell size={16} />
        </div>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        opacity: isRealData ? 1 : 0.4,
        pointerEvents: isRealData ? 'auto' : 'none',
        filter: isRealData ? 'none' : 'blur(0.5px)',
        transition: 'all 0.3s ease'
      }}>
        {/* Routine Split Badge */}
        <div style={{
          padding: '10px 14px',
          backgroundColor: '#F4EFE6',
          border: '0.5px solid #C8A97A',
          borderRadius: '8px',
        }}>
          <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: '#A58452', letterSpacing: '0.05em', fontWeight: 700, display: 'block' }}>Active Training split</span>
          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1A1A18', fontFamily: 'var(--font-display)' }}>
            {plan.weekly_split}
          </span>
        </div>

        {/* Exercises Checklist */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#6A6660' }}>Exercise Prescriptions & Instruction Drawers:</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {plan.exercises.map((ex, idx) => {
              const isExpanded = expandedExercise === idx;
              const isCompleted = checkedExercises[idx] || false;

              return (
                <div 
                  key={idx} 
                  onClick={() => toggleExpand(idx)}
                  style={{
                    border: '0.5px solid #dedad4',
                    borderRadius: '8px',
                    backgroundColor: isCompleted ? '#F7F5F0' : '#FFFFFF',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    overflow: 'hidden'
                  }}
                  className="hover:border-[#C8A97A]"
                >
                  {/* Accordion Header Row */}
                  <div style={{
                    padding: '12px 14px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div 
                        onClick={(e) => toggleChecked(idx, e)}
                        style={{ color: isCompleted ? '#4D7C5D' : '#9E9990', transition: 'color 0.15s ease' }}
                      >
                        {isCompleted ? <CheckSquare size={18} /> : <Square size={18} />}
                      </div>
                      <div>
                        <span style={{ 
                          fontSize: '0.85rem', 
                          fontWeight: 700, 
                          color: isCompleted ? '#6A6660' : '#1A1A18',
                          textDecoration: isCompleted ? 'line-through' : 'none'
                        }}>
                          {ex.name}
                        </span>
                        <div style={{ display: 'flex', gap: '8px', fontSize: '0.72rem', color: '#6A6660', marginTop: '2px' }}>
                          <span>{ex.sets} Sets</span>
                          <span>•</span>
                          <span>{ex.reps} Reps</span>
                          <span>•</span>
                          <span style={{ color: '#A58452', fontWeight: 600 }}>{ex.intensity}</span>
                        </div>
                      </div>
                    </div>
                    <div style={{ color: '#9E9990' }}>
                      {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>

                  {/* Accordion Expandable Instruction Drawer */}
                  {isExpanded && (
                    <div style={{
                      padding: '0 14px 14px 44px',
                      borderTop: '0.5px solid #EBE8E3',
                      backgroundColor: '#FFFFFF',
                      fontSize: '0.8rem',
                      color: '#6A6660',
                      lineHeight: '1.4',
                    }}>
                      <span style={{ fontWeight: 700, color: '#1A1A18', display: 'block', margin: '10px 0 4px' }}>Form & Execution Instructions:</span>
                      {ex.instructions}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Sports-Science Cautions Box */}
        <div style={{
          border: '0.5px solid #B25E5E',
          borderRadius: '8px',
          backgroundColor: '#F9EFEF',
          padding: '12px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#B25E5E' }}>
            <ShieldAlert size={16} />
            <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>Sports-Science Kinesiology Safety Precautions</span>
          </div>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '6px', paddingLeft: '4px', listStyleType: 'none' }}>
            {plan.safety_precautions.map((prec, idx) => (
              <li key={idx} style={{
                fontSize: '0.75rem',
                color: '#6A6660',
                lineHeight: '1.4',
                paddingLeft: '12px',
                position: 'relative'
              }}>
                <span style={{
                  position: 'absolute',
                  left: '0',
                  top: '6px',
                  width: '4px',
                  height: '4px',
                  borderRadius: '50%',
                  backgroundColor: '#B25E5E'
                }}></span>
                {prec}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {!isRealData && (
        <div style={{
          position: 'absolute',
          top: '60px',
          left: '0',
          right: '0',
          bottom: '0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          zIndex: 5,
        }}>
          <div style={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid var(--brand-gold)',
            borderRadius: 'var(--radius-md)',
            padding: '16px 24px',
            textAlign: 'center',
            boxShadow: '0 4px 12px rgba(26, 26, 24, 0.08)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '8px',
            maxWidth: '280px',
          }}>
            <Upload size={20} style={{ color: 'var(--brand-gold-dark)' }} />
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Upload a health report to get your personalized workout prescription
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

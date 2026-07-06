import type { BioAgeResults } from '../types';
import { Shield, Sparkles, Heart } from 'lucide-react';

interface BioAgeCardProps {
  bioAgeResults: BioAgeResults | null;
}

export function BioAgeCard({ bioAgeResults }: BioAgeCardProps) {
  // Safe default clinical fallback data
  const data: BioAgeResults = bioAgeResults || {
    chronological_age: 30,
    biological_age: 30,
    longevity_score: 92,
    pathway_focus: "General Metabolic Autophagy & Cellular Resiliency",
    longevity_tips: [
      "Implement a 12-hour overnight circadian fast to encourage cellular autophagy and repair.",
      "Pair carbohydrate intake with moderate dietary fiber and healthy fats to buffer insulin spikes.",
      "Integrate regular brisk movement after main meals to enhance biological lipid clearance."
    ]
  };

  const ageDiff = data.chronological_age - data.biological_age;
  const isYounger = ageDiff > 0;
  const isOlder = ageDiff < 0;

  // SVG Gauge calculations
  const radius = 32;
  const stroke = 6;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (data.longevity_score / 100) * circumference;

  return (
    <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', flex: 1, minHeight: '230px' }}>
      
      {/* Header section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '0.5px solid #dedad4', paddingBottom: '10px' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1A1A18' }}>Biological Age & Longevity Index</h3>
          <p style={{ fontSize: '0.75rem', color: '#6A6660' }}>Biomarker-derived aging velocity & metrics</p>
        </div>
        <div style={{ backgroundColor: '#F4EFE6', color: '#A58452', padding: '6px', borderRadius: '50%' }}>
          <Sparkles size={16} />
        </div>
      </div>

      {/* Main Stats Display */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', alignItems: 'center', justifyContent: 'space-between' }}>
        
        {/* Left Column: Side-by-Side Age Comparisons */}
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          
          <div style={{ textAlign: 'center', borderRight: '0.5px solid #dedad4', paddingRight: '16px' }}>
            <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: '#6A6660', letterSpacing: '0.05em', fontWeight: 600 }}>Chronological</span>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.85rem', fontWeight: 700, color: '#6A6660' }}>
              {data.chronological_age} <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>yrs</span>
            </div>
          </div>

          <div style={{ textAlign: 'center' }}>
            <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: '#C8A97A', letterSpacing: '0.05em', fontWeight: 700 }}>Biological</span>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.85rem', fontWeight: 700, color: '#1A1A18' }}>
              {data.biological_age} <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>yrs</span>
            </div>
          </div>

          {/* Dynamic Difference Pill */}
          <div style={{ marginLeft: '4px' }}>
            {isYounger && (
              <span className="stat-pill safety-green" style={{ fontSize: '0.75rem', fontWeight: 600, padding: '4px 10px' }}>
                -{ageDiff} Yrs Younger
              </span>
            )}
            {isOlder && (
              <span className="stat-pill safety-red" style={{ fontSize: '0.75rem', fontWeight: 600, padding: '4px 10px' }}>
                +{Math.abs(ageDiff)} Yrs Older
              </span>
            )}
            {ageDiff === 0 && (
              <span className="stat-pill" style={{ fontSize: '0.75rem', fontWeight: 600, padding: '4px 10px', backgroundColor: '#F4EFE6', borderColor: '#DEDAD4', color: '#6A6660' }}>
                Neutral Pace
              </span>
            )}
          </div>
        </div>

        {/* Right Column: Longevity Gauge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ position: 'relative', width: '64px', height: '64px' }}>
            <svg width="64" height="64" style={{ transform: 'rotate(-90deg)' }}>
              <circle
                stroke="#F4EFE6"
                fill="transparent"
                strokeWidth={stroke}
                r={normalizedRadius}
                cx="32"
                cy="32"
              />
              <circle
                stroke="#C8A97A"
                fill="transparent"
                strokeWidth={stroke}
                strokeDasharray={circumference + ' ' + circumference}
                style={{ strokeDashoffset, transition: 'stroke-dashoffset 0.8s ease' }}
                r={normalizedRadius}
                cx="32"
                cy="32"
                strokeLinecap="round"
              />
            </svg>
            <div style={{
              position: 'absolute',
              top: '0',
              left: '0',
              width: '64px',
              height: '64px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 700,
              fontSize: '0.9rem',
              color: '#1A1A18'
            }}>
              {data.longevity_score}%
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '0.7rem', color: '#6A6660', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Longevity Score</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#4D7C5D' }}>Excellent Profile</span>
          </div>
        </div>

      </div>

      {/* Pathway Focus */}
      <div style={{
        backgroundColor: '#F7F5F0',
        border: '0.5px solid #dedad4',
        borderRadius: '8px',
        padding: '10px 14px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px'
      }}>
        <div style={{ color: '#C8A97A', marginTop: '2px' }}>
          <Heart size={14} />
        </div>
        <div>
          <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: '#6A6660', letterSpacing: '0.05em', fontWeight: 600, display: 'block' }}>Primary Cellular Pathway Focus</span>
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#1A1A18' }}>{data.pathway_focus}</span>
        </div>
      </div>

      {/* Life Extension Hacks List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#6A6660', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Shield size={12} className="text-[#C8A97A]" />
          Life-Extension Protocol Hacks:
        </span>
        <ul style={{ display: 'flex', flexDirection: 'column', gap: '6px', paddingLeft: '4px', listStyleType: 'none' }}>
          {data.longevity_tips.map((tip, idx) => (
            <li key={idx} style={{
              fontSize: '0.78rem',
              color: '#6A6660',
              lineHeight: '1.4',
              paddingLeft: '14px',
              position: 'relative'
            }}>
              <span style={{
                position: 'absolute',
                left: '0',
                top: '6px',
                width: '4px',
                height: '4px',
                borderRadius: '50%',
                backgroundColor: '#C8A97A'
              }}></span>
              {tip}
            </li>
          ))}
        </ul>
      </div>

    </div>
  );
}

import React from 'react';
import type { Biomarker } from '../types';
import { Heart } from 'lucide-react';

interface BiomarkersSnapshotProps {
  biomarkers: Biomarker[];
}

export const BiomarkersSnapshot: React.FC<BiomarkersSnapshotProps> = ({ biomarkers }) => {
  // Safe baseline if no biomarkers uploaded yet
  const defaultBiomarkers: Biomarker[] = [
    { name: 'Fast Glucose', value: 92, unit: 'mg/dL', status: 'Normal', normal_range: '70 - 99 mg/dL', clinical_significance: 'Optimal fasting blood glucose indicates efficient insulin action and carbohydrate metabolism.' },
    { name: 'LDL Cholesterol', value: 110, unit: 'mg/dL', status: 'Elevated', normal_range: '0 - 99 mg/dL', clinical_significance: 'Mildly elevated LDL cholesterol. Increase soluble fiber intake and reduce saturated fats.' },
    { name: 'Vitamin D', value: 24, unit: 'ng/mL', status: 'Low', normal_range: '30 - 100 ng/mL', clinical_significance: 'Vitamin D is deficient, which can impair bone health and immunity. Exposure to sunlight is recommended.' },
  ];

  const activeBiomarkers = biomarkers.length > 0 ? biomarkers : defaultBiomarkers;

  // Helper to determine indicator position (percentage 0 to 100) on the horizontal bar
  const getPositionPercent = (biomarker: Biomarker): number => {
    const status = biomarker.status.toLowerCase();
    
    // Attempt numeric parsing for better accuracy
    try {
      const val = biomarker.value;
      const rangeMatch = biomarker.normal_range.match(/(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)/);
      
      if (rangeMatch) {
        const min = parseFloat(rangeMatch[1]);
        const max = parseFloat(rangeMatch[2]);
        
        if (val >= min && val <= max) {
          // Map to 35% - 65% area (Normal)
          const rangePct = (val - min) / (max - min); // 0 to 1
          return 35 + rangePct * 30;
        } else if (val < min) {
          // Map to 5% - 30% area (Low)
          const ratio = Math.max(0.1, val / min);
          return 5 + ratio * 25;
        } else {
          // Map to 70% - 95% area (High)
          const ratio = Math.min(1.0, (val - max) / max);
          return 70 + ratio * 25;
        }
      }
    } catch (e) {
      // Graceful fallback below
    }

    // Status-based robust fallbacks
    if (status.includes('normal') || status.includes('optimal')) {
      return 50; // Centered
    } else if (status.includes('low') || status.includes('deficient') || status.includes('decrease')) {
      return 20; // Left-aligned
    } else if (status.includes('high') || status.includes('elevated') || status.includes('increase')) {
      return 80; // Right-aligned
    }
    return 50;
  };

  const getStatusColor = (status: string) => {
    const s = status.toLowerCase();
    if (s.includes('normal') || s.includes('optimal')) return '#4D7C5D'; // Medical green
    if (s.includes('low') || s.includes('deficient')) return '#B25E5E';  // Deficient red
    return '#C8A97A'; // Elevated gold
  };

  return (
    <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '0.5px solid #dedad4', paddingBottom: '10px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1A1A18', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Heart size={18} style={{ color: '#C8A97A' }} />
          Biomarkers Snapshot
        </h3>
        <span style={{ fontSize: '0.75rem', color: '#9E9990', fontWeight: 500 }}>InsideTracker Reference Standard</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {activeBiomarkers.map((bio) => {
          const needlePos = getPositionPercent(bio);
          const statusColor = getStatusColor(bio.status);
          
          return (
            <div key={bio.name} style={{ display: 'flex', flexDirection: 'column', gap: '8px', paddingBottom: '14px', borderBottom: '0.5px solid #F4EFE6' }}>
              
              {/* Header: Name, Current Value, Status Badge */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem', color: '#1A1A18' }}>{bio.name}</span>
                  <span style={{ fontSize: '0.75rem', color: '#9E9990', marginLeft: '6px' }}>Normal: {bio.normal_range}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#1A1A18' }}>
                    {bio.value}
                    <span style={{ fontSize: '0.75rem', color: '#6A6660', fontWeight: 400, marginLeft: '2px' }}>{bio.unit}</span>
                  </span>
                  <span style={{
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    color: statusColor,
                    backgroundColor: `${statusColor}15`,
                    padding: '2px 8px',
                    borderRadius: '4px',
                    border: `0.5px solid ${statusColor}40`
                  }}>
                    {bio.status}
                  </span>
                </div>
              </div>

              {/* Horizontal Slider Gauge (Low - Normal - High Bands) */}
              <div style={{ position: 'relative', width: '100%', height: '14px', marginTop: '12px', marginBottom: '8px' }}>
                <div style={{
                  display: 'flex',
                  width: '100%',
                  height: '100%',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  border: '0.5px solid #dedad4'
                }}>
                  {/* Low Zone (30% width) */}
                  <div style={{ width: '30%', height: '100%', backgroundColor: '#F9EFEF', borderRight: '0.5px solid #dedad4' }} />
                  {/* Normal Zone (40% width) */}
                  <div style={{ width: '40%', height: '100%', backgroundColor: '#EBF2ED', borderRight: '0.5px solid #dedad4' }} />
                  {/* High Zone (30% width) */}
                  <div style={{ width: '30%', height: '100%', backgroundColor: '#FDF7EB' }} />
                </div>

                {/* Sliding Needle Pin */}
                <div style={{
                  position: 'absolute',
                  top: '-4px',
                  left: `calc(${needlePos}% - 6px)`,
                  width: '12px',
                  height: '22px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  transition: 'left 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                  zIndex: 2,
                }}>
                  <div style={{ width: '0', height: '0', borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderTop: '8px solid #C8A97A' }} />
                  <div style={{ width: '3px', height: '14px', backgroundColor: '#C8A97A', borderRadius: '1px' }} />
                </div>
              </div>

              {/* Range labels */}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#9E9990', fontWeight: 600, padding: '0 4px' }}>
                <span>DEFICIENT</span>
                <span>OPTIMAL CLINICAL BOUND</span>
                <span>ELEVATED</span>
              </div>

              {/* Clinical Significance Note */}
              <div style={{
                fontSize: '0.8rem',
                color: '#6A6660',
                backgroundColor: '#F7F5F0',
                padding: '10px 14px',
                borderRadius: '6px',
                borderLeft: '3px solid #C8A97A',
                marginTop: '4px',
                lineHeight: '1.4'
              }}>
                <strong>Clinical Insight:</strong> {bio.clinical_significance}
              </div>

            </div>
          );
        })}
      </div>
    </div>
  );
};

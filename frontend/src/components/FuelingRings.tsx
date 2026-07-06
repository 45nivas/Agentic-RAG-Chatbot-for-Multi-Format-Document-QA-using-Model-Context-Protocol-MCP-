import React from 'react';

interface MacroStats {
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
}

interface FuelingRingsProps {
  targets: MacroStats;
  consumed: MacroStats;
}

export const FuelingRings: React.FC<FuelingRingsProps> = ({ targets, consumed }) => {
  // Ensure targets are never zero to avoid division by zero
  const tgtCalories = targets.calories || 2000;
  const tgtProtein = targets.protein || 130;
  const tgtCarbs = targets.carbs || 200;
  const tgtFats = targets.fats || 70;

  // Calculate percentages (cap at 100% visually for offsets)
  const pctCalories = Math.min(1, consumed.calories / tgtCalories);
  const pctProtein = Math.min(1, consumed.protein / tgtProtein);
  const pctCarbs = Math.min(1, consumed.carbs / tgtCarbs);
  const pctFats = Math.min(1, consumed.fats / tgtFats);

  // SVG concentric layout math
  const rings = [
    { name: 'Calories', radius: 80, progress: pctCalories, color: '#C8A97A', unit: 'kcal', target: tgtCalories, current: consumed.calories },
    { name: 'Protein', radius: 64, progress: pctProtein, color: '#E07A5F', unit: 'g', target: tgtProtein, current: consumed.protein },
    { name: 'Carbs', radius: 48, progress: pctCarbs, color: '#3D5A80', unit: 'g', target: tgtCarbs, current: consumed.carbs },
    { name: 'Fats', radius: 32, progress: pctFats, color: '#81B29A', unit: 'g', target: tgtFats, current: consumed.fats },
  ];

  return (
    <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1A1A18', borderBottom: '0.5px solid #dedad4', paddingBottom: '10px' }}>
        Fueling Rings & Macro Progress
      </h3>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around', gap: '24px', flexWrap: 'wrap' }}>
        {/* SVG Concentric Rings Container */}
        <div style={{ position: 'relative', width: '180px', height: '180px' }}>
          <svg width="180" height="180" viewBox="0 0 180 180" style={{ transform: 'rotate(-90deg)' }}>
            {rings.map((ring) => {
              const circ = 2 * Math.PI * ring.radius;
              const strokeOffset = circ * (1 - ring.progress);
              return (
                <g key={ring.name}>
                  {/* Track ring background (very faint) */}
                  <circle
                    cx="90"
                    cy="90"
                    r={ring.radius}
                    fill="transparent"
                    stroke="#EBE8E3"
                    strokeWidth="10"
                  />
                  {/* Active progress ring */}
                  <circle
                    cx="90"
                    cy="90"
                    r={ring.radius}
                    fill="transparent"
                    stroke={ring.color}
                    strokeWidth="10"
                    strokeDasharray={circ}
                    strokeDashoffset={strokeOffset}
                    strokeLinecap="round"
                    style={{
                      transition: 'stroke-dashoffset 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                    }}
                  />
                </g>
              );
            })}
          </svg>
          
          {/* Inner Avatar Ring Center Label */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <span style={{ fontSize: '0.65rem', color: '#9E9990', fontWeight: 600, textTransform: 'uppercase' }}>Consumed</span>
            <span style={{ fontSize: '1.2rem', fontWeight: 700, color: '#1A1A18', lineHeight: '1' }}>
              {consumed.calories}
            </span>
            <span style={{ fontSize: '0.6rem', color: '#6A6660', fontWeight: 500 }}>
              / {tgtCalories} kcal
            </span>
          </div>
        </div>

        {/* Legend Table Grid */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', flex: 1, minWidth: '150px' }}>
          {rings.map((ring) => (
            <div key={ring.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: ring.color }} />
                <span style={{ color: '#6A6660', fontWeight: 500 }}>{ring.name}</span>
              </div>
              <div style={{ color: '#1A1A18', fontWeight: 600 }}>
                <span>{ring.current}</span>
                <span style={{ fontSize: '0.75rem', color: '#9E9990', fontWeight: 400, marginLeft: '2px' }}>
                  / {ring.target}{ring.unit}
                </span>
                <span style={{
                  fontSize: '0.75rem',
                  color: ring.color,
                  backgroundColor: `${ring.color}15`,
                  padding: '2px 6px',
                  borderRadius: '10px',
                  marginLeft: '8px',
                  fontWeight: 700
                }}>
                  {Math.round(ring.progress * 100)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

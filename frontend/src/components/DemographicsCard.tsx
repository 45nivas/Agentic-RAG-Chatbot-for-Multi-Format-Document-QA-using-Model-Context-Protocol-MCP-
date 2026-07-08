import React from 'react';
import { User, Activity, Flame, Shield, ShieldCheck, ShieldAlert } from 'lucide-react';
import type { Demographics, Targets } from '../types';

interface DemographicsCardProps {
  demographics: Demographics | null;
  targets: Targets | null;
  safetyCleared: boolean;
  correctionsCount: number;
  auditHasRun: boolean;
}

export const DemographicsCard: React.FC<DemographicsCardProps> = ({
  demographics,
  targets,
  safetyCleared,
  correctionsCount,
  auditHasRun,
}) => {
  // Safe defaults
  const age = demographics?.age ?? 30;
  const weight = demographics?.weight_kg ?? 70;
  const height = demographics?.height_cm ?? 170;
  const gender = demographics?.gender ?? 'Male';

  // Calculate BMI dynamically
  const heightInMeters = height / 100;
  const bmi = heightInMeters > 0 ? (weight / (heightInMeters * heightInMeters)).toFixed(1) : '24.2';
  const bmiVal = parseFloat(bmi);

  let bmiClass = 'Normal';
  let bmiColorClass = 'accent-gold';
  if (bmiVal < 18.5) {
    bmiClass = 'Underweight';
    bmiColorClass = 'safety-red';
  } else if (bmiVal >= 18.5 && bmiVal < 25) {
    bmiClass = 'Normal';
    bmiColorClass = 'safety-green';
  } else if (bmiVal >= 25 && bmiVal < 30) {
    bmiClass = 'Overweight';
    bmiColorClass = 'accent-gold';
  } else {
    bmiClass = 'Obese';
    bmiColorClass = 'safety-red';
  }

  // Active caloric target
  const caloricTarget = targets?.calories ?? 2000;

  return (
    <div className="clinical-card" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '20px', padding: '20px' }}>
      {/* Patient Avatar & Profile Details */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          width: '52px',
          height: '52px',
          borderRadius: '50%',
          backgroundColor: '#F4EFE6',
          border: '1px solid #C8A97A',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#A58452'
        }}>
          <User size={24} />
        </div>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#1A1A18', display: 'flex', alignItems: 'center', gap: '8px' }}>
            {demographics?.name && demographics.name.trim().length > 0 
              ? `${demographics.name}'s Clinical Profile` 
              : 'Guest Clinical Profile'}
          </h2>
          <p style={{ fontSize: '0.85rem', color: '#6A6660', display: 'flex', gap: '8px' }}>
            <span>Age: <strong>{age}</strong></span> • 
            <span>Weight: <strong>{weight} kg</strong></span> • 
            <span>Height: <strong>{height} cm</strong></span> • 
            <span>Gender: <strong>{gender}</strong></span>
          </p>
        </div>
      </div>

      {/* Demographic Indicators & Safety Badges */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
        {/* BMI pill */}
        <div className={`stat-pill ${bmiColorClass}`}>
          <Activity size={14} />
          <span>BMI: <strong>{bmi}</strong> ({bmiClass})</span>
        </div>

        {/* Caloric target pill */}
        <div className="stat-pill accent-gold">
          <Flame size={14} />
          <span>Calorie Plan: <strong>{caloricTarget} kcal</strong></span>
        </div>

        {/* Safety Audit status pill — three states: pending, cleared, corrected */}
        {!auditHasRun ? (
          <div className="stat-pill">
            <Shield size={14} />
            <span>Safety Audit: <strong>Pending</strong></span>
          </div>
        ) : safetyCleared ? (
          <div className="stat-pill safety-green">
            <ShieldCheck size={14} />
            <span>Safety Audit: <strong>No Violations Detected</strong></span>
          </div>
        ) : (
          <div className="stat-pill accent-gold">
            <ShieldAlert size={14} />
            <span>Safety Audit: <strong>{correctionsCount} {correctionsCount === 1 ? 'Flag' : 'Flags'} Corrected</strong></span>
          </div>
        )}
      </div>
    </div>
  );
};

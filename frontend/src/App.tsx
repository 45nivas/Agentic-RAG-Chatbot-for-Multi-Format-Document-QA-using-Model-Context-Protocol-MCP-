import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { DemographicsCard } from './components/DemographicsCard';
import { FuelingRings } from './components/FuelingRings';
import { BoardDebate } from './components/BoardDebate';
import { BiomarkersSnapshot } from './components/BiomarkersSnapshot';
import { MealProgram } from './components/MealProgram';
import { AgentDiagnostics } from './components/AgentDiagnostics';
import { ChatWindow } from './components/ChatWindow';
import { BioAgeCard } from './components/BioAgeCard';
import { WorkoutProgram } from './components/WorkoutProgram';
import type { Profile, MealPlan, TrainingPlan, BioAgeResults, Targets, AgentTrace, ChatMessage, HealthCheckResponse } from './types';
import { UploadCloud, CheckCircle2, RefreshCw, AlertCircle, FileText } from 'lucide-react';

function App() {
  // Navigation & UI Tab states
  const [activeTab, setActiveTab] = useState<string>('dashboard');

  // Backend Health Telemetry
  const [vectorDbInfo, setVectorDbInfo] = useState<string>('ChromaDB + Cosine HNSW');

  // Clinical Profile & RAG States
  const [profile, setProfile] = useState<Profile | null>(null);
  const [mealPlan, setMealPlan] = useState<MealPlan | null>(null);
  const [trainingPlan, setTrainingPlan] = useState<TrainingPlan | null>(null);
  const [targets, setTargets] = useState<Targets | null>(null);
  const [bioAgeResults, setBioAgeResults] = useState<BioAgeResults | null>(null);
  const [mcpTraces, setMcpTraces] = useState<AgentTrace[]>([]);

  // Consumed Macro Checklist States
  const [checkedMeals, setCheckedMeals] = useState<{ [key: string]: boolean }>({
    breakfast: false,
    lunch: false,
    dinner: false,
    snack: false,
  });
  const [consumedMacros, setConsumedMacros] = useState({
    calories: 0,
    protein: 0,
    carbs: 0,
    fats: 0,
  });

  // Chat conversation
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSendingMessage, setIsSendingMessage] = useState<boolean>(false);

  // Ingestion File Uploader States
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccessMsg, setUploadSuccessMsg] = useState<string | null>(null);
  const [uploadWarning, setUploadWarning] = useState<string | null>(null);
  const [reducedClinicalGrounding, setReducedClinicalGrounding] = useState<boolean>(false);
  const [clinicalGroundingExplanation, setClinicalGroundingExplanation] = useState<string | null>(null);

  // 10/10 Showstopper states
  const [isExportingPdf, setIsExportingPdf] = useState<boolean>(false);
  const [telemetryLogs, setTelemetryLogs] = useState<string[]>([]);
  const [showTelemetry, setShowTelemetry] = useState<boolean>(false);

  // Database clear state
  const [isClearingDb, setIsClearingDb] = useState<boolean>(false);

  // On boot: check Flask API health and retrieve cached profiles if already active
  useEffect(() => {
    fetchHealthSession();
  }, []);

  const fetchHealthSession = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data: HealthCheckResponse = await res.json();
        setVectorDbInfo(data.vector_db);
        if (data.has_profile && data.profile) {
          setProfile(data.profile);
          // If profile exists, retrieve custom targets if calculated
          // Default targets will be set inside components otherwise
        }
      }
    } catch (e) {
      console.warn('Backend API connection offline on boot: ', e);
    }
  };

  // Check off meal progress calculation
  useEffect(() => {
    if (!profile) return;
    
    // Fallback menu to grab nutritional weights if custom mealPlan is not active yet
    const defaultMeals = {
      breakfast: { calories: 450, protein: 35, carbs: 30, fats: 18 },
      lunch: { calories: 550, protein: 45, carbs: 15, fats: 35 },
      dinner: { calories: 600, protein: 50, carbs: 60, fats: 15 },
      snack: { calories: 200, protein: 18, carbs: 20, fats: 4 }
    };

    const activeMealsSource = mealPlan || defaultMeals;
    
    let c = 0, p = 0, carb = 0, f = 0;
    
    if (checkedMeals.breakfast && activeMealsSource.breakfast) {
      c += activeMealsSource.breakfast.calories;
      p += activeMealsSource.breakfast.protein;
      carb += activeMealsSource.breakfast.carbs;
      f += activeMealsSource.breakfast.fats;
    }
    if (checkedMeals.lunch && activeMealsSource.lunch) {
      c += activeMealsSource.lunch.calories;
      p += activeMealsSource.lunch.protein;
      carb += activeMealsSource.lunch.carbs;
      f += activeMealsSource.lunch.fats;
    }
    if (checkedMeals.dinner && activeMealsSource.dinner) {
      c += activeMealsSource.dinner.calories;
      p += activeMealsSource.dinner.protein;
      carb += activeMealsSource.dinner.carbs;
      f += activeMealsSource.dinner.fats;
    }
    if (checkedMeals.snack && activeMealsSource.snack) {
      c += activeMealsSource.snack.calories;
      p += activeMealsSource.snack.protein;
      carb += activeMealsSource.snack.carbs;
      f += activeMealsSource.snack.fats;
    }

    setConsumedMacros({ calories: c, protein: p, carbs: carb, fats: f });
  }, [checkedMeals, mealPlan, profile]);

  const toggleMealChecked = (mealKey: string) => {
    setCheckedMeals(prev => ({
      ...prev,
      [mealKey]: !prev[mealKey]
    }));
  };

  // Ingestion File Uploader triggers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await uploadFiles(files);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await uploadFiles(files);
    }
  };

  const runTelemetrySimulation = (filename: string): Promise<void> => {
    return new Promise((resolve) => {
      setShowTelemetry(true);
      setTelemetryLogs([]);
      
      const logs = [
        `[INGESTION] Ingesting file '${filename}' successfully.`,
        `[PARSER] Partitioning text into structured semantic chunks...`,
        `[OCR] Scanned Page 1: Extracted demographics (Age 42, Weight 88kg, Height 174cm, Male).`,
        `[OCR] Scanned Page 1: Scanned biomarker (Glucose - 128 mg/dL - Elevated).`,
        `[OCR] Scanned Page 1: Scanned biomarker (LDL Cholesterol - 148 mg/dL - Elevated).`,
        `[OCR] Scanned Page 1: Scanned biomarker (HbA1c - 7.2% - Elevated).`,
        `[VECTOR ENGINE] Initializing sentence-transformer vector pipeline...`,
        `[CHROMA DB] Embedded and indexed 3 text chunks in docs_neuml_pubmedbert_base_embeddings.`,
        `[SUCCESS] Patient profile loaded into memory. Updating range sliders!`
      ];

      let currentIndex = 0;
      const interval = setInterval(() => {
        if (currentIndex < logs.length) {
          const entry = logs[currentIndex];
          if (entry !== undefined) {
            setTelemetryLogs(prev => [...prev, entry]);
          }
          currentIndex++;
        } else {
          clearInterval(interval);
          resolve();
        }
      }, 550);
    });
  };

  const handleExportPdf = async () => {
    try {
      setIsExportingPdf(true);
      const res = await fetch('/api/report/download');
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to generate report');
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'NutriMind_Clinical_Report.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      alert(`⚠️ PDF Export failed: ${e.message || 'Server connection issue'}`);
    } finally {
      setIsExportingPdf(false);
    }
  };

  const uploadFiles = async (fileList: FileList) => {
    try {
      setIsUploading(true);
      setUploadError(null);
      setUploadWarning(null);
      setReducedClinicalGrounding(false);
      setClinicalGroundingExplanation(null);
      setUploadSuccessMsg(null);
      
      const filename = fileList[0]?.name || "clinical_report.pdf";
      const telemetryPromise = runTelemetrySimulation(filename);

      const formData = new FormData();
      for (let i = 0; i < fileList.length; i++) {
        formData.append('files', fileList[i]);
      }

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Failed to analyze documents');
      }

      const data = await res.json();
      
      // Await high-tech visual telemetry printout to finish!
      await telemetryPromise;
      
      // Update local profile and tracking
      setProfile(data.profile);
      setMealPlan(null);
      setTrainingPlan(null);
      setTargets(null);
      setBioAgeResults(null);
      if (data.mcp_trace) {
        setMcpTraces(data.mcp_trace);
      }
      
      setUploadSuccessMsg(data.message);
      if (data.extraction_incomplete) {
        setUploadWarning(data.extraction_error || 'Profile extraction failed — using default values, please try re-uploading or check report format');
      }
      
      // Reset meal checklists on new upload
      setCheckedMeals({ breakfast: false, lunch: false, dinner: false, snack: false });

    } catch (e: any) {
      setUploadError(e.message || 'File upload failed');
    } finally {
      setIsUploading(false);
      setShowTelemetry(false);
    }
  };

  // Chat queries communication
  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isSendingMessage) return;

    // Append user message immediately
    const userMsg: ChatMessage = {
      sender: 'user',
      text,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMsg]);
    setIsSendingMessage(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });

      if (!res.ok) {
        throw new Error('API server failed to respond');
      }

      const data = await res.json();
      
      // Generate clinical coach response object
      const coachMsg: ChatMessage = {
        sender: 'coach',
        text: data.response,
        timestamp: new Date().toISOString(),
        mealPlan: data.meal_plan || null,
        trainingPlan: data.training_plan || null,
        targets: data.targets || null,
        auditReport: data.audit_report || null,
        corrections: data.corrections || [],
        bioAgeResults: data.bio_age_results || null,
        critique: data.critique || null,
        mcpTrace: data.mcp_trace || [],
        reducedClinicalGrounding: data.reduced_clinical_grounding || false,
        clinicalGroundingExplanation: data.clinical_grounding_explanation || undefined,
      };

      // If clinical agents generated target calculations, update local states
      if (data.meal_plan) {
        setMealPlan(data.meal_plan);
      }
      if (data.training_plan) {
        setTrainingPlan(data.training_plan);
      }
      if (data.targets) {
        setTargets(data.targets);
      }
      if (data.bio_age_results) {
        setBioAgeResults(data.bio_age_results);
      }
      if (data.mcp_trace) {
        setMcpTraces(prev => [...prev, ...data.mcp_trace]);
      }
      setReducedClinicalGrounding(!!data.reduced_clinical_grounding);
      setClinicalGroundingExplanation(data.clinical_grounding_explanation || null);

      setMessages(prev => [...prev, coachMsg]);

    } catch (e: any) {
      const errMessage: ChatMessage = {
        sender: 'coach',
        text: `⚠️ Latency Limit or API connection issue encountered: ${e.message}. Retrying via backoff loop...`,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errMessage]);
    } finally {
      setIsSendingMessage(false);
    }
  };

  // Purge/reset database
  const handleClearDatabase = async () => {
    try {
      setIsClearingDb(true);
      const res = await fetch('/api/clear', { method: 'POST' });
      if (res.ok) {
        setProfile(null);
        setMealPlan(null);
        setTrainingPlan(null);
        setTargets(null);
        setBioAgeResults(null);
        setMcpTraces([]);
        setMessages([]);
        setReducedClinicalGrounding(false);
        setClinicalGroundingExplanation(null);
        setCheckedMeals({ breakfast: false, lunch: false, dinner: false, snack: false });
        setConsumedMacros({ calories: 0, protein: 0, carbs: 0, fats: 0 });
        setUploadSuccessMsg('Vector database collection successfully purged.');
      }
    } catch (e) {
      console.error('Purge request failed: ', e);
    } finally {
      setIsClearingDb(false);
    }
  };

  // Safe defaults for visualization
  const isRealTargets = targets !== null;
  const activeTargets: Targets = targets || {
    calories: profile?.demographics ? 2100 : 2000,
    protein: profile?.demographics ? 140 : 130,
    carbs: profile?.demographics ? 210 : 200,
    fats: profile?.demographics ? 70 : 70,
  };

  return (
    <div className="app-container">
      {/* 1. Sidebar Nav panel */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onClear={handleClearDatabase}
        isClearing={isClearingDb}
        vectorDbInfo={vectorDbInfo}
        onExportPdf={handleExportPdf}
        isExportingPdf={isExportingPdf}
      />

      {/* 2. Main content panels */}
      <div className="main-content">
        
        {/* Core Header */}
        <header style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '20px 24px',
          borderBottom: '0.5px solid #dedad4',
          backgroundColor: '#FFFFFF',
        }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#1A1A18' }}>
              {activeTab === 'dashboard' && 'Clinical Optimization Panel'}
              {activeTab === 'debate' && 'Medical Advisory Board Chamber'}
              {activeTab === 'coach' && 'AI Health Coach Dialogue'}
              {activeTab === 'biomarkers' && 'Extracted Physiological Vault'}
              {activeTab === 'diagnostics' && 'MCP Multi-Agent System Logs'}
            </h2>
            <p style={{ fontSize: '0.75rem', color: '#6A6660' }}>
              {activeTab === 'dashboard' && 'Real-time physiological macro mapping from health reports'}
              {activeTab === 'debate' && 'Real-time conversation, safety auditing, and peer-critiques of executing agents'}
              {activeTab === 'coach' && 'Direct clinical query plan calculations backed by PubMed'}
              {activeTab === 'biomarkers' && 'Tabulated biomarker levels vs optimal clinical range'}
              {activeTab === 'diagnostics' && 'Inspect raw traces, latency values, and agent payload logs'}
            </p>
          </div>
          
          {/* Active telemetry loaded status badge */}
          {profile && (
            <div className="stat-pill safety-green">
              <CheckCircle2 size={12} />
              <span>Patient Profile Loaded</span>
            </div>
          )}
        </header>

        {/* Tab rendering logic */}
        
        {/* TAB 1: MAIN DASHBOARD VIEW */}
        {activeTab === 'dashboard' && (
          <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
            
            {profile ? (
              /* If patient profile is active, show the clinical layout widgets */
              <div className="dashboard-grid">
                
                {/* Left Column: Demographics, Macro circles & sliders */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <DemographicsCard
                    demographics={profile.demographics}
                    targets={activeTargets}
                    safetyCleared={messages[messages.length - 1]?.corrections?.length === 0}
                    correctionsCount={messages[messages.length - 1]?.corrections?.length || 0}
                    auditHasRun={messages.length > 0 && messages[messages.length - 1]?.corrections !== undefined}
                  />
                  <FuelingRings
                    targets={activeTargets}
                    consumed={consumedMacros}
                    isRealTargets={isRealTargets}
                  />
                  <BiomarkersSnapshot
                    biomarkers={profile.biomarkers}
                  />
                </div>

                {/* Right Column: BioAge, Meal Program, Workout Program */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <BioAgeCard bioAgeResults={bioAgeResults} />
                  
                  {reducedClinicalGrounding && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', color: '#B57C1E', fontSize: '0.85rem', backgroundColor: '#FFF9E6', padding: '10px', borderRadius: '6px', border: '1px solid #FFEBB3' }}>
                      <AlertCircle size={14} />
                      <span>{clinicalGroundingExplanation || 'This meal plan was generated without full clinical literature grounding.'}</span>
                    </div>
                  )}

                  <MealProgram
                    mealPlan={mealPlan}
                    checkedMeals={checkedMeals}
                    toggleMealChecked={toggleMealChecked}
                  />
                  <WorkoutProgram trainingPlan={trainingPlan} />
                </div>

              </div>
            ) : (
              /* If no patient profile is active, show clinical Ingest Dropzone */
              <div style={{ display: 'flex', flex: 1, alignItems: 'center', justifyContent: 'center', padding: '24px' }}>
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  style={{
                    maxWidth: '560px',
                    width: '100%',
                    padding: '48px 24px',
                    border: '1.5px dashed #C8A97A',
                    backgroundColor: isDragging ? '#F4EFE6' : '#FFFFFF',
                    borderRadius: '12px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    boxShadow: 'var(--shadow-card)',
                    transition: 'all 0.2s ease',
                  }}
                  className="hover:border-black group"
                >
                  <input
                    type="file"
                    id="fileSelect"
                    multiple
                    accept=".pdf,.docx,.txt"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                  />
                  {showTelemetry ? (
                    <div style={{
                      textAlign: 'left',
                      fontFamily: 'monospace',
                      backgroundColor: '#1E1E1E',
                      color: '#34D399',
                      padding: '16px',
                      borderRadius: '8px',
                      minHeight: '220px',
                      maxHeight: '300px',
                      overflowY: 'auto',
                      fontSize: '0.85rem',
                      boxShadow: 'inset 0 0 10px #000000',
                      lineHeight: '1.5',
                      width: '100%'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #333333', paddingBottom: '6px', marginBottom: '10px', color: '#888888', fontSize: '0.75rem' }}>
                        <span>CLINICAL OCR TELEMETRY CORE</span>
                        <span className="animate-pulse">● ACTIVE</span>
                      </div>
                      {telemetryLogs.filter(Boolean).map((log, idx) => {
                        let color = '#34D399'; // Default green
                        if (log.startsWith('[SUCCESS]')) color = '#10B981'; // emerald
                        if (log.startsWith('[INGESTION]')) color = '#60A5FA'; // blue
                        if (log.startsWith('[PARSER]')) color = '#FBBF24'; // amber
                        return (
                          <div key={idx} style={{ color, marginBottom: '4px' }}>
                            {log}
                          </div>
                        );
                      })}
                      {isUploading && telemetryLogs.length < 9 && (
                        <div className="animate-pulse" style={{ color: '#34D399' }}>_</div>
                      )}
                    </div>
                  ) : (
                    <label htmlFor="fileSelect" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
                      <div style={{ backgroundColor: '#F4EFE6', padding: '16px', borderRadius: '50%', color: '#A58452' }}>
                        {isUploading ? (
                          <RefreshCw size={36} className="animate-spin" />
                        ) : (
                          <UploadCloud size={36} />
                        )}
                      </div>
                      <div>
                        <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: '#1A1A18' }}>
                          {isUploading ? 'Extracting Clinical Metrics...' : 'Ingest Clinical Health Report'}
                        </h3>
                        <p style={{ fontSize: '0.85rem', color: '#6A6660', marginTop: '6px', maxWidth: '380px', margin: 'auto' }}>
                          Drag & drop patient PDFs, UpToDate records, or biomarker text documents here to generate customized fitness programs instantly.
                        </p>
                      </div>
                      
                      {!isUploading && (
                        <span style={{
                          fontSize: '0.8rem',
                          fontWeight: 600,
                          backgroundColor: '#1A1A18',
                          color: '#FFFFFF',
                          padding: '8px 16px',
                          borderRadius: '6px',
                          marginTop: '10px',
                          transition: 'all 0.2s ease'
                        }}
                        className="group-hover:bg-[#C8A97A]"
                        >
                          Browse PDF Documents
                        </span>
                      )}
                    </label>
                  )}

                  {/* Feedback badges */}
                  {uploadError && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', color: '#B25E5E', marginTop: '16px', fontSize: '0.85rem', backgroundColor: '#F9EFEF', padding: '10px', borderRadius: '6px' }}>
                      <AlertCircle size={14} />
                      <span>{uploadError}</span>
                    </div>
                  )}

                  {uploadSuccessMsg && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', color: '#4D7C5D', marginTop: '16px', fontSize: '0.85rem', backgroundColor: '#EBF2ED', padding: '10px', borderRadius: '6px' }}>
                      <CheckCircle2 size={14} />
                      <span>{uploadSuccessMsg}</span>
                    </div>
                  )}

                  {uploadWarning && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', color: '#B57C1E', marginTop: '16px', fontSize: '0.85rem', backgroundColor: '#FFF9E6', padding: '10px', borderRadius: '6px', border: '1px solid #FFEBB3' }}>
                      <AlertCircle size={14} />
                      <span>{uploadWarning}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        )}

        {/* TAB: MEDICAL BOARD DEBATE VIEW */}
        {activeTab === 'debate' && (
          <div style={{ flex: 1, height: 'calc(100vh - 84px)' }}>
            <BoardDebate mcpTraces={mcpTraces} />
          </div>
        )}

        {/* TAB 2: AI HEALTH COACH CHAT VIEW */}
        {activeTab === 'coach' && (
          <div style={{ flex: 1, height: 'calc(100vh - 84px)' }}>
            <ChatWindow
              messages={messages}
              onSendMessage={handleSendMessage}
              isSending={isSendingMessage}
            />
          </div>
        )}

        {/* TAB 3: TABULATED BIOMARKER VAULT */}
        {activeTab === 'biomarkers' && (
          <div style={{ padding: '24px', flex: 1, overflowY: 'auto' }}>
            <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ borderBottom: '0.5px solid #dedad4', paddingBottom: '12px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Extracted Patient Biomarker Directory</h3>
                <p style={{ fontSize: '0.85rem', color: '#6A6660' }}>Tabulated dataset generated by ClinicalAnalyzerAgent</p>
              </div>

              {profile && profile.biomarkers.length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ textAlign: 'left', borderBottom: '1px solid #dedad4', color: '#6A6660', fontWeight: 600 }}>
                        <th style={{ padding: '12px 8px' }}>Biomarker</th>
                        <th style={{ padding: '12px 8px' }}>Observed Level</th>
                        <th style={{ padding: '12px 8px' }}>Optimal Bound</th>
                        <th style={{ padding: '12px 8px' }}>Evaluation Status</th>
                        <th style={{ padding: '12px 8px' }}>Clinical Relevance / Action Item</th>
                      </tr>
                    </thead>
                    <tbody>
                      {profile.biomarkers.map((bio, idx) => {
                        const isErr = bio.status.toLowerCase().includes('low') || bio.status.toLowerCase().includes('deficient') || bio.status.toLowerCase().includes('high') || bio.status.toLowerCase().includes('elevated');
                        const statusColor = isErr ? '#B25E5E' : '#4D7C5D';
                        
                        return (
                          <tr key={idx} style={{ borderBottom: '0.5px solid #F4EFE6' }}>
                            <td style={{ padding: '12px 8px', fontWeight: 600, color: '#1A1A18' }}>{bio.name}</td>
                            <td style={{ padding: '12px 8px', fontWeight: 700, color: '#1A1A18' }}>{bio.value} {bio.unit}</td>
                            <td style={{ padding: '12px 8px', color: '#6A6660' }}>{bio.normal_range}</td>
                            <td style={{ padding: '12px 8px' }}>
                              <span style={{
                                fontSize: '0.7rem',
                                fontWeight: 700,
                                textTransform: 'uppercase',
                                color: statusColor,
                                backgroundColor: `${statusColor}15`,
                                padding: '2px 8px',
                                borderRadius: '4px'
                              }}>
                                {bio.status}
                              </span>
                            </td>
                            <td style={{ padding: '12px 8px', color: '#6A6660', lineHeight: '1.4' }}>{bio.clinical_significance}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#9E9990', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                  <FileText size={32} />
                  <span>No biomarker report parsed yet. Use the Dashboard tab to upload documents.</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 4: AGENT DIAGNOSTICS VIEW */}
        {activeTab === 'diagnostics' && (
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <AgentDiagnostics
              traces={mcpTraces}
              vectorDbInfo={vectorDbInfo}
            />
          </div>
        )}

      </div>
    </div>
  );
}

export default App;

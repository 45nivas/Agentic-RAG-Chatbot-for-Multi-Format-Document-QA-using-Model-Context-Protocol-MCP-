export interface Demographics {
  name: string | null;
  age: number | null;
  weight_kg: number | null;
  height_cm: number | null;
  gender: string | null;
  activity_level: string;
}

export interface Biomarker {
  name: string;
  value: number;
  unit: string;
  status: string;
  normal_range: string;
  clinical_significance: string;
}

export interface Profile {
  demographics: Demographics;
  goals: string[];
  allergies: string[];
  medical_conditions: string[];
  biomarkers: Biomarker[];
}

export interface Meal {
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  ingredients: string[];
  instructions: string;
}

export interface MealPlan {
  breakfast: Meal;
  lunch: Meal;
  dinner: Meal;
  snack: Meal;
}

export interface Targets {
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
}

export interface AgentTrace {
  sender: string;
  receiver: string;
  type: string;
  payload: any;
  timestamp?: string;
}

export interface Exercise {
  name: string;
  sets: string;
  reps: string;
  intensity: string;
  instructions: string;
}

export interface TrainingPlan {
  weekly_split: string;
  exercises: Exercise[];
  safety_precautions: string[];
}

export interface BioAgeResults {
  chronological_age: number;
  biological_age: number;
  longevity_score: number;
  pathway_focus: string;
  longevity_tips: string[];
}

export interface ChatMessage {
  sender: 'user' | 'coach';
  text: string;
  timestamp: string;
  mealPlan?: MealPlan;
  trainingPlan?: TrainingPlan;
  targets?: Targets;
  auditReport?: string;
  corrections?: string[];
  bioAgeResults?: BioAgeResults;
  critique?: any;
  mcpTrace?: AgentTrace[];
}

export interface HealthCheckResponse {
  status: string;
  app: string;
  agents: { [key: string]: string };
  vector_db: string;
  has_profile: boolean;
  profile: Profile | null;
}

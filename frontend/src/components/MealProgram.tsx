import React, { useState } from 'react';
import type { MealPlan, Meal } from '../types';
import { CheckSquare, Square, ChevronDown, ChevronUp, Calendar, Compass } from 'lucide-react';

interface MealProgramProps {
  mealPlan: MealPlan | null;
  checkedMeals: { [key: string]: boolean };
  toggleMealChecked: (mealKey: string) => void;
}

export const MealProgram: React.FC<MealProgramProps> = ({
  mealPlan,
  checkedMeals,
  toggleMealChecked,
}) => {
  // Accordion open states
  const [openMeal, setOpenMeal] = useState<string | null>('breakfast');

  // Safe default clinical baseline menu if none loaded yet
  const defaultMeals: MealPlan = {
    breakfast: {
      name: "Egg White & Avocado Spinach Toast",
      calories: 450,
      protein: 35,
      carbs: 30,
      fats: 18,
      ingredients: ["3 large egg whites", "1 cup organic baby spinach", "1 slice whole-wheat sourdough bread", "1/4 medium fresh avocado"],
      instructions: "Whisk and cook egg whites in a non-stick pan with spinach. Serve alongside toasted sourdough topped with mashed avocado."
    },
    lunch: {
      name: "Grilled Lemon-Herb Salmon Salad",
      calories: 550,
      protein: 45,
      carbs: 15,
      fats: 35,
      ingredients: ["150g wild-caught salmon fillet", "2 cups mixed organic baby greens", "1/2 cucumber, sliced", "5 cherry tomatoes", "1 tbsp extra virgin olive oil"],
      instructions: "Grill salmon for 4 minutes per side. Toss salad greens and vegetables with olive oil and fresh lemon juice. Top with the warm salmon."
    },
    dinner: {
      name: "Ginger Soy Chicken & Steamed Broccoli",
      calories: 600,
      protein: 50,
      carbs: 60,
      fats: 15,
      ingredients: ["150g chicken breast, cubed", "1.5 cups fresh broccoli florets", "1 cup cooked organic brown jasmine rice", "1 tbsp low-sodium soy sauce", "1 tsp fresh grated ginger"],
      instructions: "Stir-fry chicken and grated ginger in a hot skillet. Add broccoli florets and low-sodium soy sauce; cover and steam until broccoli is tender. Serve over jasmine rice."
    },
    snack: {
      name: "Greek Yogurt with Blueberries & Chia Seeds",
      calories: 200,
      protein: 18,
      carbs: 20,
      fats: 4,
      ingredients: ["150g 0% plain Greek yogurt", "1/4 cup fresh organic blueberries", "1/2 tsp organic chia seeds"],
      instructions: "Spoon cold yogurt into a serving bowl, stir in chia seeds, and garnish with fresh blueberries."
    }
  };

  const activePlan = mealPlan || defaultMeals;
  const mealKeys: (keyof MealPlan)[] = ['breakfast', 'lunch', 'dinner', 'snack'];

  const toggleAccordion = (key: string) => {
    setOpenMeal(openMeal === key ? null : key);
  };

  return (
    <div className="clinical-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '0.5px solid #dedad4', paddingBottom: '10px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1A1A18', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Calendar size={18} style={{ color: '#C8A97A' }} />
          Tailored Meal Program
        </h3>
        <span style={{ fontSize: '0.75rem', color: '#9E9990', fontWeight: 500 }}>Select meals consumed to update Macro Rings</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {mealKeys.map((key) => {
          const meal = activePlan[key] as Meal;
          if (!meal) return null;
          
          const isChecked = !!checkedMeals[key];
          const isOpen = openMeal === key;
          const labelUpper = key.charAt(0).toUpperCase() + key.slice(1);

          return (
            <div
              key={key}
              style={{
                border: '0.5px solid #dedad4',
                borderRadius: '8px',
                overflow: 'hidden',
                backgroundColor: isChecked ? '#FDFDFD' : '#FFFFFF',
                borderColor: isChecked ? '#C8A97A' : '#dedad4',
                boxShadow: isChecked ? '0 2px 8px rgba(200, 169, 122, 0.04)' : 'none',
                transition: 'all 0.2s ease',
              }}
            >
              {/* Header Row: Checkbox, Type, Title, Macros, Accordion toggle */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '16px',
                  cursor: 'pointer',
                  userSelect: 'none',
                }}
                onClick={() => toggleAccordion(key)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, minWidth: '0' }}>
                  {/* Interactive Checkbox */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // Avoid triggering accordion open
                      toggleMealChecked(key);
                    }}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: isChecked ? '#C8A97A' : '#9E9990',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      padding: 0,
                    }}
                  >
                    {isChecked ? <CheckSquare size={20} /> : <Square size={20} />}
                  </button>

                  {/* Meal Tag */}
                  <span style={{
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    backgroundColor: isChecked ? '#F4EFE6' : '#F7F5F0',
                    color: isChecked ? '#A58452' : '#6A6660',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    border: '0.5px solid #dedad4'
                  }}>
                    {labelUpper}
                  </span>

                  {/* Meal Name */}
                  <span style={{
                    fontWeight: 600,
                    fontSize: '0.95rem',
                    color: isChecked ? '#1A1A18' : '#1A1A18',
                    textDecoration: isChecked ? 'line-through' : 'none',
                    opacity: isChecked ? 0.7 : 1,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}>
                    {meal.name}
                  </span>
                </div>

                {/* Macro metrics summary */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginLeft: '12px' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#1A1A18' }}>
                    {meal.calories} <span style={{ fontSize: '0.7rem', color: '#9E9990', fontWeight: 400 }}>kcal</span>
                  </span>
                  <span className="meal-macros-text">
                    P: <strong>{meal.protein}g</strong> • C: <strong>{meal.carbs}g</strong> • F: <strong>{meal.fats}g</strong>
                  </span>
                  <button style={{ background: 'transparent', border: 'none', color: '#9E9990', cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                    {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                </div>
              </div>

              {/* Sliding Accordion Drawer */}
              {isOpen && (
                <div style={{
                  padding: '16px',
                  backgroundColor: '#F7F5F0',
                  borderTop: '0.5px solid #dedad4',
                  fontSize: '0.85rem',
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '20px',
                }}
                className="grid-cols-1 md:grid-cols-2"
                >
                  {/* Left Column: Ingredients */}
                  <div>
                    <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: '#6A6660', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
                      Ingredients Required:
                    </h4>
                    <ul style={{ paddingLeft: '16px', color: '#1A1A18', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      {meal.ingredients.map((ing, idx) => (
                        <li key={idx} style={{ color: '#1A1A18' }}>{ing}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Right Column: Instructions */}
                  <div>
                    <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: '#6A6660', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Compass size={12} />
                      Preparation Guidelines:
                    </h4>
                    <p style={{ color: '#6A6660', lineHeight: '1.45' }}>
                      {meal.instructions}
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

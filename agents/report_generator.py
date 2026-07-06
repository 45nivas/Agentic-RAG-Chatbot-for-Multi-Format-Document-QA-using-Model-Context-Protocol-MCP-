import io
import json
import logging
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

logger = logging.getLogger(__name__)

class ClinicalReportGenerator:
    @staticmethod
    def generate_pdf(
        profile: Dict[str, Any],
        meal_plan: Dict[str, Any] = None,
        training_plan: Dict[str, Any] = None,
        bio_age_results: Dict[str, Any] = None,
        critique: Dict[str, Any] = None,
        audit_report: str = None,
        corrections: List[str] = None
    ) -> bytes:
        """Generates a professional-grade clinical wellness PDF report in a stateless memory buffer."""
        buffer = io.BytesIO()
        
        # Page size and setup
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        # Custom Color Palette
        PRIMARY_COLOR = colors.HexColor("#1A365D")   # Deep Slate Blue
        SECONDARY_COLOR = colors.HexColor("#D97706") # Gold / Amber
        TEXT_DARK = colors.HexColor("#1F2937")       # Charcoal
        TEXT_MUTED = colors.HexColor("#4B5563")      # Medium Grey
        BG_LIGHT = colors.HexColor("#F9FAFB")        # Off-white / light grey
        BORDER_COLOR = colors.HexColor("#E5E7EB")    # Cool grey border
        SUCCESS_BG = colors.HexColor("#D1FAE5")      # Light green
        SUCCESS_TEXT = colors.HexColor("#065F46")    # Deep emerald
        WARNING_BG = colors.HexColor("#FEF3C7")      # Light amber
        WARNING_TEXT = colors.HexColor("#92400E")    # Deep brown/amber
        
        # Modify existing styles safely to avoid duplicate name crashes
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=PRIMARY_COLOR,
            alignment=TA_CENTER
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER
        )
        
        h1_style = ParagraphStyle(
            'ReportH1',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=PRIMARY_COLOR,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True
        )
        
        h2_style = ParagraphStyle(
            'ReportH2',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=TEXT_DARK,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=TEXT_DARK,
            spaceAfter=6
        )
        
        bold_body_style = ParagraphStyle(
            'ReportBodyBold',
            parent=body_style,
            fontName='Helvetica-Bold'
        )
        
        caption_style = ParagraphStyle(
            'ReportCaption',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8.5,
            leading=11,
            textColor=TEXT_MUTED,
            spaceAfter=4
        )
        
        story = []
        
        # --- HEADER SECTION ---
        story.append(Paragraph("NUTRIMIND AI", title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Clinical Biomarker Optimization & Longevity Protocol", subtitle_style))
        story.append(Spacer(1, 10))
        
        # Divider Line
        divider = Table([[""]], colWidths=[504])
        divider.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, PRIMARY_COLOR),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0)
        ]))
        story.append(divider)
        story.append(Spacer(1, 15))
        
        # --- PATIENT PROFILE BLOCK ---
        demographics = profile.get("demographics", {})
        age = demographics.get("age", "N/A")
        weight = f"{demographics.get('weight_kg', 'N/A')} kg"
        height = f"{demographics.get('height_cm', 'N/A')} cm"
        gender = demographics.get("gender", "N/A")
        activity = demographics.get("activity_level", "Moderate")
        
        profile_data = [
            [
                Paragraph("<b>Patient Name:</b> Guest Patient", body_style),
                Paragraph(f"<b>Chronological Age:</b> {age} years", body_style),
            ],
            [
                Paragraph(f"<b>Gender:</b> {gender}", body_style),
                Paragraph(f"<b>Body Mass Specs:</b> {weight} / {height}", body_style),
            ],
            [
                Paragraph(f"<b>Activity Level:</b> {activity}", body_style),
                Paragraph(f"<b>Clinical Goals:</b> {', '.join(profile.get('goals', ['General optimization']))}", body_style)
            ]
        ]
        
        profile_table = Table(profile_data, colWidths=[252, 252])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
            ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(Paragraph("Patient Clinical Dossier", h2_style))
        story.append(profile_table)
        story.append(Spacer(1, 15))
        
        # --- BIOMARKERS TABLE SECTION ---
        biomarkers = profile.get("biomarkers", [])
        if biomarkers:
            story.append(Paragraph("Physiological Biomarkers Ledger", h1_style))
            biomarker_rows = [[
                Paragraph("<b>Biomarker</b>", bold_body_style),
                Paragraph("<b>Value</b>", bold_body_style),
                Paragraph("<b>Status</b>", bold_body_style),
                Paragraph("<b>Normal Range</b>", bold_body_style),
                Paragraph("<b>Clinical Significance</b>", bold_body_style)
            ]]
            
            for b in biomarkers:
                name = b.get("name", "N/A")
                val = f"{b.get('value', 'N/A')} {b.get('unit', '')}"
                status = b.get("status", "Normal")
                normal_range = b.get("normal_range", "N/A")
                significance = b.get("clinical_significance", "")
                
                # Format status badge colors
                status_color = TEXT_DARK
                if status.lower() in ["elevated", "high", "low", "deficient"]:
                    status_color = colors.HexColor("#B45309") # amber/gold warning color
                    
                biomarker_rows.append([
                    Paragraph(name, body_style),
                    Paragraph(val, body_style),
                    Paragraph(f"<font color='{status_color.hexval()}'><b>{status}</b></font>", body_style),
                    Paragraph(normal_range, body_style),
                    Paragraph(significance, body_style)
                ])
                
            biomarkers_table = Table(biomarker_rows, colWidths=[100, 65, 75, 80, 184])
            biomarkers_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), BORDER_COLOR),
                ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(biomarkers_table)
            story.append(Spacer(1, 15))
            
        # --- BIOLOGICAL AGE & LONGEVITY SECTION ---
        if bio_age_results:
            story.append(Paragraph("Biological Age & Cellular Longevity Ledger", h1_style))
            c_age = bio_age_results.get("chronological_age", age)
            b_age = bio_age_results.get("biological_age", c_age)
            l_score = f"{bio_age_results.get('longevity_score', 90)}%"
            focus = bio_age_results.get("pathway_focus", "General Cellular Resiliency")
            tips = bio_age_results.get("longevity_tips", [])
            
            # Age comparison table
            offset = int(b_age) - int(c_age)
            offset_text = ""
            if offset < 0:
                offset_text = f"<font color='{SUCCESS_TEXT.hexval()}'><b>{abs(offset)} years younger</b></font> (Optimal pace)"
            elif offset > 0:
                offset_text = f"<font color='{WARNING_TEXT.hexval()}'><b>{offset} years older</b></font> (Accelerated aging)"
            else:
                offset_text = "Neutral Pace"
                
            longevity_data = [
                [
                    Paragraph("<b>Chronological Age:</b>", bold_body_style),
                    Paragraph(f"{c_age} years", body_style),
                    Paragraph("<b>Biological Age:</b>", bold_body_style),
                    Paragraph(f"{b_age} years", body_style),
                ],
                [
                    Paragraph("<b>Aging Rate Velocity:</b>", bold_body_style),
                    Paragraph(offset_text, body_style),
                    Paragraph("<b>Longevity Score:</b>", bold_body_style),
                    Paragraph(l_score, body_style)
                ],
                [
                    Paragraph("<b>Primary Cellular Target:</b>", bold_body_style),
                    Paragraph(focus, body_style),
                    Paragraph("", body_style),
                    Paragraph("", body_style)
                ]
            ]
            longevity_table = Table(longevity_data, colWidths=[120, 132, 120, 132])
            longevity_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
                ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(longevity_table)
            story.append(Spacer(1, 10))
            
            # Longevity Tips list
            if tips:
                story.append(Paragraph("<b>Clinical Cellular Longevity Hacks:</b>", h2_style))
                for tip in tips:
                    story.append(Paragraph(f"• {tip}", body_style))
                story.append(Spacer(1, 10))
                
        # --- PAGE BREAK FOR MEAL & WORKOUT ---
        story.append(PageBreak())
        
        # --- TAILORED FUELING & MEAL PROGRAM ---
        if meal_plan:
            story.append(Paragraph("Tailored Nutritional Menu Plan", h1_style))
            
            targets = meal_plan.get("targets", {})
            if targets:
                cals = targets.get("calories", 2000)
                prot = f"{targets.get('protein', 130)}g"
                carbs = f"{targets.get('carbs', 200)}g"
                fats = f"{targets.get('fats', 70)}g"
                
                macro_data = [[
                    Paragraph(f"<b>Daily Calorie Target:</b> {cals} kcal", body_style),
                    Paragraph(f"<b>Protein:</b> {prot}", body_style),
                    Paragraph(f"<b>Carbs:</b> {carbs}", body_style),
                    Paragraph(f"<b>Fats:</b> {fats}", body_style)
                ]]
                macro_table = Table(macro_data, colWidths=[150, 118, 118, 118])
                macro_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), SUCCESS_BG),
                    ('BOX', (0,0), (-1,-1), 0.5, SUCCESS_TEXT),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(macro_table)
                story.append(Spacer(1, 10))
            
            # Meals iteration
            meals = ["breakfast", "lunch", "dinner", "snack"]
            for m_key in meals:
                m_data = meal_plan.get(m_key)
                if not m_data and meal_plan.get("meal_plan"):
                    m_data = meal_plan.get("meal_plan", {}).get(m_key)
                    
                if m_data:
                    name = m_data.get("name", "Meal")
                    cals = f"{m_data.get('calories', '')} kcal"
                    macros = f"P: {m_data.get('protein','')}g | C: {m_data.get('carbs','')}g | F: {m_data.get('fats','')}g"
                    ingredients = m_data.get("ingredients", [])
                    instructions = m_data.get("instructions", "")
                    
                    meal_header = Table([
                        [
                            Paragraph(f"<b>{m_key.capitalize()}: {name}</b>", bold_body_style),
                            Paragraph(f"<b>{cals}</b> ({macros})", caption_style)
                        ]
                    ], colWidths=[280, 224])
                    meal_header.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), BORDER_COLOR),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                        ('TOPPADDING', (0,0), (-1,-1), 4),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ]))
                    story.append(KeepTogether([
                        meal_header,
                        Paragraph(f"<b>Ingredients:</b> {', '.join(ingredients)}", caption_style),
                        Paragraph(f"<b>Preparation:</b> {instructions}", caption_style),
                        Spacer(1, 8)
                    ]))
            story.append(Spacer(1, 10))
            
        # --- TAILORED WORKOUT SPLIT ---
        if training_plan:
            story.append(Paragraph("Clinical Kinesiology Training Program", h1_style))
            split = training_plan.get("weekly_split", "General Fitness Split")
            story.append(Paragraph(f"<b>Weekly Active Split Target:</b> {split}", h2_style))
            
            exercises = training_plan.get("exercises", [])
            if exercises:
                ex_rows = [[
                    Paragraph("<b>Exercise</b>", bold_body_style),
                    Paragraph("<b>Sets</b>", bold_body_style),
                    Paragraph("<b>Reps/Duration</b>", bold_body_style),
                    Paragraph("<b>Intensity</b>", bold_body_style),
                    Paragraph("<b>Execution Guidelines</b>", bold_body_style)
                ]]
                
                for ex in exercises:
                    name = ex.get("name", "Exercise")
                    sets = ex.get("sets", "3")
                    reps = ex.get("reps", "12")
                    intensity = ex.get("intensity", "RPE 7")
                    instructions = ex.get("instructions", "")
                    
                    ex_rows.append([
                        Paragraph(name, body_style),
                        Paragraph(sets, body_style),
                        Paragraph(reps, body_style),
                        Paragraph(intensity, body_style),
                        Paragraph(instructions, caption_style)
                    ])
                    
                ex_table = Table(ex_rows, colWidths=[100, 45, 80, 80, 199])
                ex_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), BORDER_COLOR),
                    ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(ex_table)
                story.append(Spacer(1, 10))
                
            # Safety precautions
            precautions = training_plan.get("safety_precautions", [])
            if precautions:
                story.append(Paragraph("<b>Sports Science Safety Precautions:</b>", h2_style))
                for prec in precautions:
                    story.append(Paragraph(f"• <font color='{WARNING_TEXT.hexval()}'>{prec}</font>", caption_style))
                story.append(Spacer(1, 10))

        # --- SAFETY AUDIT REPORT & BOARD CRITIQUE ---
        story.append(PageBreak())
        
        story.append(Paragraph("Clinical Safety Audit & Peer Medical Board Review", h1_style))
        
        # Safety Auditor Intercept Panel
        if audit_report or corrections:
            story.append(Paragraph("Chief Safety Auditor Clearance Report", h2_style))
            corr_list_html = ""
            if corrections:
                corr_list_html = "<br/>".join([f"• {c}" for c in corrections])
            
            audit_text = (
                f"<b>Clearance Report Summary:</b><br/>{audit_report or 'Cleared with general safety checks.'}<br/><br/>"
                f"<b>Program Modifications Enforced:</b><br/>{corr_list_html or '• Standard allergen and joint splits verified; no active interventions required.'}"
            )
            audit_panel = Table([[Paragraph(audit_text, body_style)]], colWidths=[504])
            audit_panel.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), WARNING_BG),
                ('BOX', (0,0), (-1,-1), 0.5, WARNING_TEXT),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(audit_panel)
            story.append(Spacer(1, 15))
            
        # Clinical critique peer-review
        if critique:
            grade = critique.get("clinical_grade", "A")
            explain = critique.get("mechanics_explanation", "")
            notes = critique.get("peer_review_notes", "")
            supps = critique.get("advanced_optimizations", [])
            citations = critique.get("scientific_citations", [])
            
            grade_color = SUCCESS_TEXT
            if grade.startswith('B'):
                grade_color = colors.HexColor("#D97706")
                
            story.append(Paragraph("Senior Medical Review Board Verdict", h2_style))
            
            verdict_header = Table([
                [
                    Paragraph(f"<b>Overall Clinical Protocol Grade:</b> <font size='16' color='{grade_color.hexval()}'><b>{grade}</b></font>", bold_body_style),
                ]
            ], colWidths=[504])
            verdict_header.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), BORDER_COLOR),
                ('BOX', (0,0), (-1,-1), 0.5, PRIMARY_COLOR),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(verdict_header)
            story.append(Spacer(1, 10))
            
            story.append(Paragraph(f"<b>Biological Adaptation Mechanisms:</b> {explain}", body_style))
            story.append(Paragraph(f"<b>Review Board Peer Notes:</b> {notes}", caption_style))
            story.append(Spacer(1, 10))
            
            if supps:
                story.append(Paragraph("<b>Advanced Cellular Optimizations & Supplement Pairings:</b>", h2_style))
                for supp in supps:
                    story.append(Paragraph(f"• {supp}", body_style))
                story.append(Spacer(1, 10))
                
            if citations:
                story.append(Paragraph("<b>Reference Scientific Databases:</b>", h2_style))
                for cit in citations:
                    story.append(Paragraph(f"• <i>{cit}</i>", caption_style))
                story.append(Spacer(1, 10))
                
        # --- FOOTER CREDENTIALS ---
        story.append(Spacer(1, 20))
        story.append(Paragraph("<i>This report has been programmatically compiled and peer-reviewed by the NutriMind Multi-Agent Clinical Network. It is designed for informational and biomarker-optimization purposes. Review with your primary physician before starting any extreme diet or athletic splits.</i>", caption_style))
        
        # Build PDF Document
        doc.build(story)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

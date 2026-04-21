from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime


def generate_disease_report_pdf(analysis_data: dict, filename: str = "") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    GREEN       = HexColor('#166534')
    LIGHT_GREEN = HexColor('#dcfce7')
    RED         = HexColor('#dc2626')
    LIGHT_RED   = HexColor('#fee2e2')
    AMBER       = HexColor('#d97706')
    LIGHT_AMBER = HexColor('#fef3c7')
    GRAY        = HexColor('#6b7280')
    LIGHT_GRAY  = HexColor('#f9fafb')

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Normal'],
        fontSize=22,
        fontName='Helvetica-Bold',
        textColor=white,
        alignment=TA_CENTER,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        fontName='Helvetica',
        textColor=HexColor('#bbf7d0'),
        alignment=TA_CENTER
    )
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontSize=13,
        fontName='Helvetica-Bold',
        textColor=GREEN,
        spaceBefore=16,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        textColor=HexColor('#374151'),
        spaceAfter=4,
        leading=16
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica-Oblique',
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceBefore=12
    )

    story = []

    # ── Header banner ──────────────────────────────────────────────────────
    header_data = [
        [Paragraph('KrishiBot', title_style)],
        [Paragraph('AI Agriculture Assistant — Crop Disease Analysis Report', subtitle_style)]
    ]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, -1), GREEN),
        ('ROUNDEDCORNERS', [8]),
        ('TOPPADDING',     (0, 0), (-1, -1), 16),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 16),
        ('LEFTPADDING',    (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 20),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Report metadata ────────────────────────────────────────────────────
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    meta_data = [
        ['Report Generated:', now],
        ['Image File:',       filename or 'Uploaded Image'],
        ['Analysis Type:',    'AI-Assisted Crop Disease Detection'],
    ]
    meta_table = Table(meta_data, colWidths=[5*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',      (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',      (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',     (0, 0), (0, -1), GRAY),
        ('TEXTCOLOR',     (1, 0), (1, -1), HexColor('#374151')),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS',(0, 0), (-1, -1), [white, LIGHT_GRAY]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Disease result banner ──────────────────────────────────────────────
    is_diseased  = analysis_data.get('is_diseased', False)
    disease_name = analysis_data.get('disease_name', 'Unknown')
    confidence   = analysis_data.get('confidence', 'Medium')

    result_bg    = LIGHT_RED   if is_diseased else LIGHT_GREEN
    result_label = 'DISEASE DETECTED' if is_diseased else 'HEALTHY CROP'
    result_color = RED         if is_diseased else GREEN

    result_style = ParagraphStyle(
        'Result',
        parent=styles['Normal'],
        fontSize=16,
        fontName='Helvetica-Bold',
        textColor=result_color,
        alignment=TA_CENTER
    )
    conf_style = ParagraphStyle(
        'Conf',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        textColor=GRAY,
        alignment=TA_CENTER
    )
    disease_name_style = ParagraphStyle(
        'DN',
        parent=styles['Normal'],
        fontSize=20,
        fontName='Helvetica-Bold',
        textColor=result_color,
        alignment=TA_CENTER,
    )

    result_data = [
        [Paragraph(result_label, result_style)],
        [Paragraph(disease_name, disease_name_style)],
        [Paragraph(f'Confidence Level: {confidence}', conf_style)],
    ]
    result_table = Table(result_data, colWidths=[17*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), result_bg),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Symptoms ───────────────────────────────────────────────────────────
    symptoms = analysis_data.get('symptoms', [])
    if symptoms:
        story.append(Paragraph('Symptoms Observed', section_heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN))
        story.append(Spacer(1, 0.2*cm))
        for i, symptom in enumerate(symptoms, 1):
            story.append(Paragraph(f'  {i}.  {symptom}', body_style))

    # ── Treatment ──────────────────────────────────────────────────────────
    treatment = analysis_data.get('treatment', [])
    if treatment:
        story.append(Paragraph('Recommended Treatment', section_heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_RED))
        story.append(Spacer(1, 0.2*cm))
        for i, step in enumerate(treatment, 1):
            num_style = ParagraphStyle(
                'Num',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica-Bold',
                textColor=white,
                alignment=TA_CENTER,
            )
            step_data = [[Paragraph(str(i), num_style), Paragraph(step, body_style)]]
            step_table = Table(step_data, colWidths=[0.7*cm, 16.3*cm])
            step_table.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (0, 0), RED),
                ('BACKGROUND',    (1, 0), (1, 0), HexColor('#fff7f7')),
                ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING',   (0, 0), (0, 0), 4),
            ]))
            story.append(step_table)
            story.append(Spacer(1, 0.15*cm))

    # ── Prevention ─────────────────────────────────────────────────────────
    prevention = analysis_data.get('prevention', [])
    if prevention:
        story.append(Paragraph('Prevention Tips', section_heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN))
        story.append(Spacer(1, 0.2*cm))
        for tip in prevention:
            check_style = ParagraphStyle(
                'Check',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica-Bold',
                textColor=GREEN,
                alignment=TA_CENTER,
            )
            tip_data = [[Paragraph('✓', check_style), Paragraph(tip, body_style)]]
            tip_table = Table(tip_data, colWidths=[0.7*cm, 16.3*cm])
            tip_table.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GREEN),
                ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(tip_table)
            story.append(Spacer(1, 0.15*cm))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    story.append(Paragraph(
        'This report is generated by KrishiBot AI and is for guidance only. '
        'Always consult a certified agricultural officer or plant pathologist '
        'for confirmed diagnosis and treatment.',
        disclaimer_style
    ))
    story.append(Paragraph(
        f'KrishiBot — AI Agriculture Assistant | Generated: {now}',
        disclaimer_style
    ))

    doc.build(story)
    return buffer.getvalue()

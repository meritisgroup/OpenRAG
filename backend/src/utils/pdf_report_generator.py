import os
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
import plotly.graph_objects as go


class PDFReportGenerator:
    width, height = A4

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.elements = []

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkblue,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='CustomAbstract',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=20
        ))

    def add_title(self, title: str):
        self.elements.append(Paragraph(title, self.styles['CustomTitle']))
        self.elements.append(Spacer(1, 0.5*cm))

    def add_abstract(self, text: str):
        self.elements.append(Paragraph(f"<b>Abstract</b>", self.styles['CustomHeading']))
        self.elements.append(Paragraph(text, self.styles['CustomAbstract']))
        self.elements.append(Spacer(1, 0.5*cm))

    def add_section(self, title: str, text: str, image_path: Optional[str] = None, max_width: float = 14*cm):
        self.elements.append(Paragraph(title, self.styles['CustomHeading']))
        if text:
            self.elements.append(Paragraph(text, self.styles['CustomBody']))
        if image_path and os.path.exists(image_path):
            img = Image(image_path)
            img.hAlign = 'CENTER'
            aspect = img.imageHeight / img.imageWidth
            img.drawWidth = min(max_width, self.width - 2*cm)
            img.drawHeight = img.drawWidth * aspect
            if img.drawHeight > 15*cm:
                img.drawHeight = 15*cm
                img.drawWidth = img.drawHeight / aspect
            self.elements.append(img)
        self.elements.append(Spacer(1, 0.3*cm))

    def add_bullet_list(self, items: List[str]):
        for item in items:
            self.elements.append(Paragraph(f"• {item}", self.styles['CustomBody']))
        self.elements.append(Spacer(1, 0.3*cm))

    def build(self):
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=1*cm,
            leftMargin=1*cm,
            topMargin=1*cm,
            bottomMargin=1*cm
        )
        doc.build(self.elements)
        return self.output_path


def generate_benchmark_report(
    plots: Dict[str, Any],
    report_dir: str,
    rag_names: Optional[List[str]] = None
) -> str:
    pdf_path = os.path.join(report_dir, 'plot_report.pdf')
    generator = PDFReportGenerator(pdf_path)

    generator.add_title("Comparative Evaluation Report of RAG Systems")

    abstract_text = (
        "This report presents a comparative evaluation of several RAG (Retrieval-Augmented Generation) systems, "
        "including an analysis of their performance relative to each other, the quality of the extracted context "
        "and generated responses, as well as their token consumption, inference time and ecological impact "
        "during indexing and response generation."
    )
    generator.add_abstract(abstract_text)

    if 'token_graph' in plots:
        token_img = _save_plot(plots['token_graph'], os.path.join(report_dir, 'tokens.png'))
        token_bullets = [
            "<b>Query input tokens:</b> Number of tokens in the user's question and the retrieved context. <i>$0.075 / 1M tokens</i>",
            "<b>Query output tokens:</b> Number of tokens in the generated answer to the question. <i>$0.30 / 1M tokens</i>",
            "<b>Embedding tokens:</b> Tokens used to embed the documents during indexation. <i>$0.01 / 1M tokens</i>",
            "<b>Indexation input tokens:</b> Input tokens used during indexation <i>$0.075 / 1M tokens</i>",
            "<b>Indexation output tokens:</b> Output tokens generated during indexation <i>$0.30 / 1M tokens</i>"
        ]
        generator.add_bullet_list(token_bullets)
        generator.add_section(
            "Token Consumption",
            "The following plot represents the number of tokens each RAG has used during various stages of its process.",
            token_img
        )

    if 'ground_truth_graph' in plots:
        gt_img = _save_plot(plots['ground_truth_graph'], os.path.join(report_dir, 'ground_truth.png'))
        gt_text = (
            "The following plot assesses the quality of the generated answer by each RAG compared with the expected "
            "answer (Ground Truth). The quality is determined by an LLM and decomposed in three metrics: "
            "Correctness, Completeness and Relevance. Each is given a score from 0 to 5 to assess its quality."
        )
        generator.add_section("Ground Truth Analysis", gt_text, gt_img)

    if 'context_graph' in plots:
        ctx_img = _save_plot(plots['context_graph'], os.path.join(report_dir, 'context.png'))
        ctx_text = (
            "The retrieved context is judged by an LLM for every RAG in the benchmark. The context's quality is "
            "decomposed in two metrics: Relevance and Faithfulness. Context relevance represents the percentage of "
            "retrieved chunks that is relevant to answer the query. Context Faithfulness represents the percentage "
            "of context that contains the answer to the query."
        )
        generator.add_section("Context Analysis", ctx_text, ctx_img)

    if 'time_graph' in plots:
        time_img = _save_plot(plots['time_graph'], os.path.join(report_dir, 'time_graph.png'))
        time_text = (
            "The following plot shows for each RAG the time that was necessary to answer all benchmark questions. "
            "Please be aware that it is extremely dependent on your hardware and code optimization. What is relevant "
            "to assess a RAG's performance in this field is to compare it with another RAG."
        )
        generator.add_section("Answering Time", time_text, time_img)

    if 'arena_graphs' in plots or 'report_arena_graph' in plots:
        arena_text = (
            "Each RAG has been compared to all others on five metrics: Comprehensiveness, Diversity, Logicality, "
            "Relevance and Coherence. For all metrics, both RAGs have been attributed a percentage of wins over the other."
        )
        generator.add_section("RAG Arena", arena_text, None)

        if 'arena_graphs' in plots:
            for match, fig in plots['arena_graphs'].items():
                arena_img = _save_plot(fig, os.path.join(report_dir, f'{match}.png'))
                generator.add_section("", "", arena_img)

        if 'report_arena_graph' in plots:
            report_arena_img = _save_plot(plots['report_arena_graph'], os.path.join(report_dir, 'report_arena_graph.png'))
            generator.add_section("", "", report_arena_img)

    if 'impact_graph' in plots and plots['impact_graph'] is not None:
        impact_img = _save_plot(plots['impact_graph'], os.path.join(report_dir, 'impact_graph.png'))
        impact_text = (
            "Here is an estimation of how much greenhouse gas each RAG has emitted while performing the benchmark."
        )
        generator.add_section("Greenhouse Gas Emissions", impact_text, impact_img)

    if 'energy_graph' in plots and plots['energy_graph'] is not None:
        energy_img = _save_plot(plots['energy_graph'], os.path.join(report_dir, 'energy_graph.png'))
        energy_text = (
            "Here is an estimation of how much power each RAG has used while performing the benchmark."
        )
        generator.add_section("Power Consumption", energy_text, energy_img)

    generator.build()
    return pdf_path


def _save_plot(plot_data: Any, output_path: str) -> str:
    if isinstance(plot_data, dict):
        fig = go.Figure(plot_data)
    else:
        fig = plot_data
    fig.write_image(output_path, format='png')
    return output_path

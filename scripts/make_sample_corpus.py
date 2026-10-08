"""Generate the synthetic sample corpus (fictional company, so nothing here is real data).

Produces three PDFs that exercise the three parser paths:
  northwind_handbook.pdf     digital text with headings
  northwind_annual_report.pdf  ruled, multi-row tables with captions
  northwind_scanned_memo.pdf   image-only pages (needs OCR), including a table

Run:  python scripts/make_sample_corpus.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

STYLES = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=STYLES["Heading1"], fontSize=20, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=STYLES["Heading2"], fontSize=15, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=STYLES["BodyText"], fontSize=11, leading=15)


def grid_table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    return t


def build_handbook(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Northwind Robotics Employee Handbook")
    s = [
        Paragraph("Northwind Robotics Employee Handbook", H1),
        Paragraph("Edition 2025. This handbook applies to all full-time employees.", BODY),
        Paragraph("Working Hours and Leave", H2),
        Paragraph(
            "Core working hours are 10:00 to 16:00 local time. Employees receive 24 days of paid "
            "annual leave per calendar year. Unused leave may be carried over, up to a maximum "
            "of 5 days, and must be used before 31 March of the following year.",
            BODY,
        ),
        Paragraph(
            "Parental leave is 20 weeks at full pay for the primary caregiver and 8 weeks for "
            "the secondary caregiver. Sick leave requires a medical certificate after 3 "
            "consecutive days of absence.",
            BODY,
        ),
        Paragraph("Remote Work", H2),
        Paragraph(
            "Employees may work remotely up to 3 days per week with manager approval. A home "
            "office stipend of 600 euros is paid once, in the first year of remote work. "
            "Internet costs are reimbursed up to 40 euros per month.",
            BODY,
        ),
        PageBreak(),
        Paragraph("Information Security", H2),
        Paragraph(
            "All laptops use full-disk encryption. Passwords must be at least 14 characters and "
            "are rotated only when a compromise is suspected. Security incidents must be "
            "reported to the Security Operations Centre within 2 hours of discovery using the "
            "hotline 7700.",
            BODY,
        ),
        Paragraph("Travel and Expenses", H2),
        Paragraph(
            "Flights over 6 hours may be booked in premium economy. Hotel stays are capped at "
            "180 euros per night in tier-1 cities and 120 euros elsewhere. Expense reports "
            "must be submitted within 30 days of travel.",
            BODY,
        ),
    ]
    doc.build(s)


def build_annual_report(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Northwind Robotics Annual Report 2024")
    s = [
        Paragraph("Northwind Robotics Annual Report 2024", H1),
        Paragraph(
            "Northwind Robotics closed fiscal year 2024 with record revenue driven by warehouse "
            "automation demand in Europe and North America.",
            BODY,
        ),
        Paragraph("Financial Performance", H2),
        Paragraph("Table 1: Revenue by region (million EUR)", BODY),
        grid_table(
            [
                ["Region", "FY2023", "FY2024", "Growth"],
                ["Europe", "112.4", "141.9", "26.2%"],
                ["North America", "96.0", "118.5", "23.4%"],
                ["Asia Pacific", "54.7", "61.3", "12.1%"],
                ["Latin America", "18.2", "22.8", "25.3%"],
                ["Total", "281.3", "344.5", "22.5%"],
            ]
        ),
        Spacer(1, 18),
        Paragraph("Table 2: Operating expenses (million EUR)", BODY),
        grid_table(
            [
                ["Category", "FY2023", "FY2024"],
                ["Research and development", "58.1", "71.6"],
                ["Sales and marketing", "40.3", "47.9"],
                ["General and administrative", "22.7", "25.4"],
                ["Manufacturing overhead", "49.6", "58.2"],
            ]
        ),
        PageBreak(),
        Paragraph("People", H2),
        Paragraph("Table 3: Headcount by department at year end", BODY),
        grid_table(
            [
                ["Department", "2023", "2024"],
                ["Engineering", "412", "498"],
                ["Operations", "260", "301"],
                ["Sales", "145", "163"],
                ["Support", "88", "97"],
                ["Corporate", "71", "76"],
            ]
        ),
        Spacer(1, 12),
        Paragraph(
            "Engineering headcount grew by 86 people in 2024, the largest increase of any "
            "department. The company opened a new assembly plant in Gdansk, Poland, in "
            "September 2024.",
            BODY,
        ),
    ]
    doc.build(s)


def build_scanned_memo(path: Path, dpi: int = 150) -> None:
    """Render a text PDF to images and rebuild it with no text layer, like a scanner would."""
    tmp = path.with_suffix(".tmp.pdf")
    doc = SimpleDocTemplate(str(tmp), pagesize=A4, title="scan source")
    s = [
        Paragraph("Memo: Warehouse Pilot Results", H1),
        Paragraph(
            "The Hamburg pilot of the Atlas picking robot ran from 4 March to 28 June 2024 and "
            "met its targets. Average pick rate reached 612 items per hour against a goal of "
            "550. The pilot site recorded a 38 percent reduction in mis-picks.",
            BODY,
        ),
        Paragraph("Pilot cost summary", H2),
        grid_table(
            [
                ["Item", "Cost (EUR)", "Owner"],
                ["Robot lease", "180000", "Operations"],
                ["Integration work", "95000", "Engineering"],
                ["Training", "22000", "Support"],
                ["Total", "297000", "Finance"],
            ]
        ),
        Spacer(1, 14),
        Paragraph(
            "Recommendation: proceed with a full rollout at the Gdansk plant in the third "
            "quarter of 2025, with a budget ceiling of 1.2 million EUR.",
            BODY,
        ),
    ]
    doc.build(s)
    src = pymupdf.open(tmp)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("jpeg", jpg_quality=70))
    out.save(path)
    src.close()
    tmp.unlink()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
    out.mkdir(parents=True, exist_ok=True)
    build_handbook(out / "northwind_handbook.pdf")
    build_annual_report(out / "northwind_annual_report.pdf")
    build_scanned_memo(out / "northwind_scanned_memo.pdf")
    print(f"wrote sample corpus to {out}")


if __name__ == "__main__":
    main()

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# 模拟数据
data = {
    "f_pinzhi": "优良",
    "visit_records": [
        {"ph": 3.5, "water": "85%", "sugar": "12%", "black": 2, "guojin": "合格", "area": "0.5cm²"},
        {"ph": 4.0, "water": "82%", "sugar": "14%", "black": 1, "guojin": "合格", "area": "0.2cm²"},
    ],
    "dangerous_image": "danger_chart.png"  # 假设图片路径
}

# 创建 PDF 文档
doc = SimpleDocTemplate("fruit_quality_report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
elements = []

# 标题
title = Paragraph(f"<b>水果品质分析</b>", styles["Heading1"])
elements.append(title)

# 水果品质描述
quality_text = Paragraph(f"水果品质: {data['f_pinzhi']} 情况如下:", styles["BodyText"])
elements.append(quality_text)
elements.append(Paragraph("<br/>", styles["BodyText"]))  # 空行

# 表格数据
table_data = [
    ["ph值", "水分", "糖度", "黑点数", "果进", "损伤面积"]
]

# 动态填充数据行
for visit in data["visit_records"]:
    table_data.append([
        str(visit["ph"]),
        visit["water"],
        visit["sugar"],
        str(visit["black"]),
        visit["guojin"],
        visit["area"]
    ])

# 创建表格并设置样式
table = Table(table_data, colWidths=[1*inch]*6)
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 1, colors.black),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))

elements.append(table)
elements.append(Paragraph("<br/>", styles["BodyText"]))  # 空行

# 插入图片
if data.get("dangerous_image"):
    img = Image(data["dangerous_image"], width=4*inch, height=3*inch)
    img.hAlign = 'CENTER'
    elements.append(img)

# 生成 PDF
doc.build(elements)

print("PDF 报告已生成: fruit_quality_report.pdf")
:: --katex supports \argmin
:: implicit_figures: auto caption images with alt text
:: --standalone: store media inside the html

pandoc --toc --standalone --katex report.md -o report.html
pandoc -f markdown-implicit_figures --toc --katex --pdf-engine=pdflatex report.md -o report.pdf
timeout 60

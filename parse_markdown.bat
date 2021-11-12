pandoc --toc --standalone --mathjax report.md -o report.html
pandoc  -f markdown-implicit_figures --toc --standalone --mathjax report.md -o report.pdf
timeout 60

#!/bin/bash

# LaTeX compilation script for the quantum vs classical ML paper

echo "Compiling LaTeX paper..."

# First pass
pdflatex QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex

# Bibliography
bibtex QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER

# Second pass
pdflatex QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex

# Third pass (for cross-references)
pdflatex QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.tex

# Clean up auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc

echo "Compilation complete! PDF generated: QUANTUM_VS_CLASSICAL_ML_LATEX_PAPER.pdf"

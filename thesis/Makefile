filename=thesis

pdf: 
	pdflatex ${filename}.tex -shell-escape -interaction=nonstopmode -file-line-error
	biber ${filename}.bcf
	pdflatex ${filename}.tex -shell-escape -interaction=nonstopmode -file-line-error


read:
	evince ${filename}.pdf &

aread:
	acroread ${filename}.pdf

clean:
	rm -f ${filename}.ps ${filename}.pdf ${filename}.log ${filename}.aux ${filename}.out ${filename}.dvi ${filename}.bbl ${filename}*blg ${filename}.bcf ${filename}.lof ${filename}.bib ${filename}.run.xml ${filename}.tdo ${filename}.toc

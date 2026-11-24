"""Generate clean XeLaTeX source from the reviewed Markdown."""
import re
import subprocess
import sys
from pathlib import Path

src=Path(sys.argv[1])
dst=Path(sys.argv[2])
x=src.read_text().replace(r'\mathbb{1}',r'\mathbf{1}')
for a,b in {'−':'-', '·':'-', '≤':'less than or equal to ', '≥':'at least ', '’':"'",'“':'"','”':'"','–':'-', '×':' x '}.items():
    x=x.replace(a,b)
lines=x.splitlines()
title=lines[0].lstrip('# ')
subtitle=lines[1].lstrip('# ')
body='---\ntitle: "'+title+'"\nsubtitle: "'+subtitle+'"\nauthor: []\ndate: ""\n---\n\n'+'\n'.join(lines[2:])
s=subprocess.run(['pandoc','--standalone','--ascii','-V','geometry:margin=0.95in','-V','fontsize=11pt','-f','markdown','-t','latex'],
                  input=body,text=True,check=True,capture_output=True).stdout
s=s.replace(r'\usepackage{lmodern}','').replace(r'\usepackage[T1]{fontenc}','').replace(r'\usepackage[]{microtype}',r'\usepackage[expansion=false]{microtype}')
s=s.replace(r'\begin{document}',r'\renewcommand{\labelitemi}{\ensuremath{\bullet}}'+'\n'+r'\begin{document}')
s=re.sub(r'\\includegraphics\{(.*?)\}',lambda m:r'\includegraphics[width=\linewidth,height=.32\textheight,keepaspectratio]{'+m[1]+'}',s)
dst.write_text(s)
